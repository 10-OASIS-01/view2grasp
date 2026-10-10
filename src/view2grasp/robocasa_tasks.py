"""Small RoboCasa tasks used to validate the mobile-manipulation loop.

This module is imported only inside the pinned RoboCasa runtime.  The task is
intentionally simpler than DeliverBrewedCoffee: a rigid can and a large bowl
remove handle and liquid semantics while retaining the essential closed loop
of grasp, held navigation, relational placement, release, and native scoring.
"""
from robocasa.environments.kitchen.kitchen import Kitchen, FixtureType, OU


class DeliverCanToBowl(Kitchen):
    """Carry one soda can from a work counter to a bowl across the kitchen."""

    EXCLUDE_LAYOUTS = Kitchen.DINING_COUNTER_EXCLUDED_LAYOUTS

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.source_counter = self.register_fixture_ref(
            "source_counter", dict(id=FixtureType.COUNTER_NON_DINING)
        )
        self.stool = self.register_fixture_ref("stool", dict(id=FixtureType.STOOL))
        self.dining_counter = self.register_fixture_ref(
            "dining_counter", dict(id=FixtureType.DINING_COUNTER, ref=self.stool)
        )
        self.init_robot_base_ref = self.source_counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "Pick up the soda can from the work counter, carry it across the "
            "kitchen, and place it inside the large bowl on the dining counter."
        )
        return ep_meta

    def _get_obj_cfgs(self):
        return [
            dict(
                name="soda_can",
                obj_groups="can",
                graspable=True,
                placement=dict(
                    fixture=self.source_counter,
                    size=(0.40, 0.30),
                    pos=(0.0, -1.0),
                ),
            ),
            dict(
                name="target_bowl",
                obj_groups="bowl",
                object_scale=1.35,
                placement=dict(
                    fixture=self.dining_counter,
                    sample_region_kwargs=dict(ref=self.stool),
                    size=(0.35, 0.35),
                    pos=("ref", "ref"),
                ),
            ),
        ]

    def _check_success(self):
        can_in_bowl = OU.check_obj_in_receptacle(
            self, "soda_can", "target_bowl", th=0.10
        )
        bowl_on_counter = OU.check_obj_fixture_contact(
            self, "target_bowl", self.dining_counter
        )
        gripper_far = OU.gripper_obj_far(self, obj_name="soda_can", th=0.15)
        return can_in_bowl and bowl_on_counter and gripper_far
