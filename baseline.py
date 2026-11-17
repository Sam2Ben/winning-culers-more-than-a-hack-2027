"""Infer team possession from tactical ATD and automatic Smart Tagging."""
import argparse
import csv
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

FPS = 25
FRAME_COUNT = 199_319
PITCH = np.array([105.3, 68.0])
GRADED_HALVES = ((11_185, 90_652), (110_808, 190_198))
LABELS = np.array(["H", "A", "D"])
OFFENSIVE_TAGS = {"BUILD UP", "PROGRESSION", "CREATING CHANCES", "BALL IN FINAL THIRD"}


def load_tracking(tactical):
    metadata = ET.parse(tactical / "BAR-ATH_tactical_atd.xml").getroot()
    home_id = metadata.find(".//Score").get("idLocalTeam")
    away_id = metadata.find(".//Score").get("idVisitingTeam")
    team_of = {p.get("id"): p.get("teamId") for p in metadata.findall(".//Players/Player")}
    channels = metadata.findall(".//DataFormatSpecification//PlayerChannelRef")
    if not channels or len(channels) % 3:
        raise ValueError("Unexpected ATD channel specification")
    track_ids = [r.get("playerChannelId").split("_")[0].removeprefix("player") for r in channels[::3]]
    groups = np.array([{home_id: 0, away_id: 1}[team_of[track]] for track in track_ids])
    names = {t.get("id"): t.findtext("Name") for t in metadata.findall(".//Teams/Team")}
    name_to_side = {names[home_id].casefold(): 0, names[away_id].casefold(): 1}
    points = np.empty((FRAME_COUNT, len(track_ids) + 1, 2), dtype=np.float32)
    row_count = 0
    with (tactical / "BAR-ATH_tactical_atd.txt").open(encoding="utf-8-sig") as stream:
        for index, line in enumerate(stream):
            frame, players, ball = line.strip().split(":")
            if index >= FRAME_COUNT or int(frame) != index + 1:
                raise ValueError("ATD rows must be consecutive and start at frame 1")
            values = np.fromstring(players.replace(";", ",") + "," + ball, sep=",")
            if values.size != points.shape[1] * 3:
                raise ValueError(f"Wrong number of tracking channels at frame {frame}")
            points[index] = values.reshape(-1, 3)[:, :2] * PITCH
            row_count += 1
    if row_count != FRAME_COUNT:
        raise ValueError(f"Expected {FRAME_COUNT} tracking rows, found {row_count}")
    return points[:, :-1], points[:, -1], groups, name_to_side


def load_tags(path, name_to_side):
    phase = np.zeros((FRAME_COUNT, 2))
    restart = np.zeros(FRAME_COUNT)
    before_restart = np.zeros(FRAME_COUNT)
    kickoffs = []
    for item in ET.parse(path).findall(".//ALL_INSTANCES/instance"):
        first = max(0, round(float(item.findtext("start")) * FPS))
        end = min(FRAME_COUNT, round(float(item.findtext("end")) * FPS))
        if end <= first:
            continue
        labels = {entry.findtext("group"): entry.findtext("text") for entry in item.findall("label")}
        side = name_to_side.get((labels.get("Team") or "").casefold())
        code = item.findtext("code")
        if code in OFFENSIVE_TAGS and side is not None:
            phase[first:end, side] = 1
        if code == "SET PIECES":
            # The tag indicates a restart window. Its boundary is approximate.
            restart[first:end] = 1
            before_restart[max(0, first - 3 * FPS):first] = 1
            if (labels.get("Type") or "").casefold() == "kick off (start)":
                kickoffs.append((first, end))
    return phase, restart, before_restart, kickoffs


def rolling_mean(values, radius):
    indices = np.arange(len(values))
    first = np.maximum(0, indices - radius)
    end = np.minimum(len(values), indices + radius + 1)
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=float)))
    return (cumulative[end] - cumulative[first]) / (end - first)


def frame_evidence(players, ball, groups, phase, restart, before_restart):
    observed = np.isfinite(ball).all(axis=1)
    displacement = np.linalg.norm(np.diff(ball, axis=0, prepend=ball[:1]), axis=1)
    velocity = FPS * displacement
    plausible = observed & (~np.isfinite(velocity) | (velocity < 45))

    gap = np.linalg.norm(players - ball[:, None, :], axis=-1)
    gap[~np.isfinite(gap)] = np.inf
    nearest = np.column_stack([gap[:, groups == side].min(axis=1) for side in (0, 1)])
    # A ball several metres away from both teams contributes little control evidence.
    proximity = np.exp(-nearest / 2.3)
    proximity[~plausible] = 0
    slow = np.where(np.isfinite(velocity), np.exp(-velocity / 12), 0.5)
    control = proximity * (0.35 + 0.65 * slow[:, None])
    margin = control[:, 0] - control[:, 1]
    signed_control = rolling_mean(margin, 5)
    tag_margin = 0.085 * (phase[:, 0] - phase[:, 1])
    team_evidence = signed_control + tag_margin
    live = 0.08 + 0.20 * np.abs(signed_control)

    boundary = np.minimum(ball, PITCH - ball).min(axis=1)
    outside = plausible & (boundary < -0.60)
    still = plausible & (velocity < 1.2)
    stationary = rolling_mean(still, 12)
    player_speed = FPS * np.linalg.norm(
        np.diff(players, axis=0, prepend=players[:1]), axis=-1
    )
    moving_tracks = np.isfinite(player_speed) & (player_speed < 12)
    slow_tracks = moving_tracks & (player_speed < 1.5)
    track_count = moving_tracks.sum(axis=1)
    quiet_players = rolling_mean(
        (track_count >= 14) & (slow_tracks.sum(axis=1) >= 0.75 * track_count), FPS
    )
    near_line = plausible & (boundary < 1.5)
    last_seen = np.maximum.accumulate(np.where(plausible, np.arange(len(ball)), -1))
    previous_near_line = (last_seen >= 0) & near_line[np.maximum(last_seen, 0)]
    sustained_missing = rolling_mean(~plausible, FPS) > 0.85
    missing_after_line = sustained_missing & previous_near_line
    restart_context = before_restart * (0.43 * (~plausible) + 0.43 * stationary)
    # A quiet group of players can occur during live play. Use it only to
    # corroborate evidence near a boundary or an automatically tagged restart.
    quiet_stoppage = quiet_players * (
        missing_after_line | ((stationary > 0.5) & near_line) | (before_restart > 0)
    )
    dead = (
        1.10 * rolling_mean(outside, 6)
        + 0.43 * stationary * near_line
        + 0.26 * missing_after_line
        + restart_context
        + 0.24 * quiet_stoppage
        - 0.24 * restart
        - 0.10 * phase.max(axis=1)
    )
    return np.column_stack((live + team_evidence, live - team_evidence, 1.5 * dead))


def decode(evidence):
    """Choose a state sequence with an explicit cost for changing state."""
    transition = np.array([[0, 8, 5], [8, 0, 5], [5, 5, 0]], dtype=float)
    score = evidence[0].copy()
    previous = np.zeros(evidence.shape, dtype=np.int8)
    states = np.arange(len(LABELS))
    for index in range(1, len(evidence)):
        candidates = score[:, None] - transition
        best = candidates.argmax(axis=0)
        previous[index] = best
        score = evidence[index] + candidates[best, states]
    path = np.empty(len(evidence), dtype=np.int8)
    path[-1] = score.argmax()
    for index in range(len(path) - 1, 0, -1):
        path[index - 1] = previous[index, path[index]]
    return LABELS[path]


def half_evidence(players, ball, groups, phase, restart, before_restart, kickoffs, first, end):
    frames = slice(first, end)
    evidence = frame_evidence(
        players[frames], ball[frames], groups, phase[frames], restart[frames], before_restart[frames]
    )
    # The kickoff tag gives an approximate lower bound for the start of play.
    kickoff = min((a for a, b in kickoffs if b > first and a < end), default=first)
    if kickoff > first:
        evidence[:kickoff - first] = [0, 0, 2]
    return evidence


def predict(data_dir):
    tactical = data_dir / "tactical"
    players, ball, groups, names = load_tracking(tactical)
    phase, restart, before_restart, kickoffs = load_tags(
        tactical / "BAR-ATH_tactical_smart_tagging.xml", names
    )
    labels = np.full(FRAME_COUNT, "X")
    for first, end in GRADED_HALVES:
        evidence = half_evidence(players, ball, groups, phase, restart, before_restart, kickoffs, first, end)
        labels[first:end] = decode(evidence)
    return labels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing tactical/")
    parser.add_argument("--output", type=Path, default=Path("Winning Culers.csv"))
    args = parser.parse_args()
    labels = predict(args.data_dir)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("frame", "label"))
        writer.writerows(enumerate(labels))
    print(f"Wrote {len(labels):,} predictions to {args.output}")


if __name__ == "__main__":
    main()
