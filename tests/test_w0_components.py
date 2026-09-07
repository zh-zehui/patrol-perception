import unittest
import numpy as np

from patrol_perception.rules.coco_discover import coco_discover_events
from patrol_perception.rules.crowd import crowd_events
from patrol_perception.rules.night_loitering import NightLoiteringRule, NightLoiteringRuleConfig
from patrol_perception.track.simple_tracker import IoUTracker
from patrol_perception.types import Detection


class TestIoUTracker(unittest.TestCase):
    def test_first_frame_assigns_track_id(self):
        tr = IoUTracker()
        dets = [Detection("person", 0.9, (0, 0, 10, 10))]
        out = tr.update(dets)
        self.assertIsNotNone(out[0].track_id)


class TestW0Rules(unittest.TestCase):
    def test_coco_discover_bicycle(self):
        dets = [Detection("bicycle", 0.8, (0, 0, 10, 10), track_id=3)]
        events = coco_discover_events(dets, active_event_ids={"bicycle"}, conf_min=0.5)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].type, "bicycle")
        self.assertEqual(events[0].track_ids, [3])

    def test_crowd_count(self):
        dets = [
            Detection("person", 0.9, (0, 0, 10, 10), track_id=i) for i in range(6)
        ]
        events = crowd_events(dets, active_event_ids={"crowd"}, conf_min=0.5, crowd_person_n=5)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].type, "crowd")

    def test_night_loitering_triggers(self):
        # Dark image -> night gate pass
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cfg = NightLoiteringRuleConfig(night_mean_threshold=60.0, history_len=20, min_dwell_frames=5, motion_max_px=1.0)
        rule = NightLoiteringRule(cfg)

        tid = 1
        d = Detection("person", 0.9, (100, 100, 200, 200), track_id=tid)
        # Keep center stable for 5 frames
        events = []
        for _ in range(5):
            events.extend(rule.update(img, [d], active_event_ids={"night_loitering"}))
        # Should emit at least one event
        self.assertTrue(any(e.type == "night_loitering" for e in events))


if __name__ == "__main__":
    unittest.main()

