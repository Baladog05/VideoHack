"""Edge-case taxonomy for autonomous-driving training data.
Each category: searches that surface candidates, and a definition the judge model uses."""

# Outdoor road cameras only (indoor warehouse / smart-space cams are out of scope for self-driving).
CAMERAS = ["pie_cam-3", "nyc_bike_gopro-1", "nyc_streets_cam-1", "nyc_streets_cam-2", "neighborhood_cam-1",
           "i24_cam-1", "sf_streets_cam-1", "sf_streets_cam-2", "sf_streets_cam-3", "sf_streets_cam-4", "sf_streets_cam-5"]

CATEGORIES = {
    "occluded_pedestrian": {
        "label": "Occluded pedestrian / jaywalker",
        "definition": "A pedestrian emerges from behind a bus, truck or parked car, or crosses mid-block outside a crosswalk.",
        "queries": ["pedestrian stepping out from behind a parked vehicle", "person jaywalking across the road mid-block",
                    "pedestrian crossing outside the crosswalk", "pedestrian appears suddenly in front of the car"]},
    "vru_violation": {
        "label": "Cyclist / scooter rule violation",
        "definition": "A cyclist, scooter or e-bike rider runs a red light, rides the wrong way, or cuts across traffic.",
        "queries": ["cyclist running a red light", "cyclist riding the wrong way against traffic",
                    "cyclist cutting across in front of the vehicle", "scooter rider weaving through traffic"]},
    "blocked_lane": {
        "label": "Blocked lane / double-parked vehicle",
        "definition": "A double-parked, broken-down or delivery vehicle blocks a lane and forces vehicles to go around it.",
        "queries": ["double parked truck blocking the lane", "delivery van stopped in the traffic lane",
                    "vehicle forced to go around a stopped truck", "construction or obstruction blocking the road"]},
    "unexpected_vehicle": {
        "label": "Unexpected vehicle manoeuvre",
        "definition": "A vehicle runs a stop sign or red light, pulls out suddenly, makes an illegal U-turn or turns across the path.",
        "queries": ["vehicle running a stop sign or red light", "car pulling out suddenly into traffic",
                    "car turning across the path of oncoming vehicle", "illegal u-turn in the middle of the road"]},
    "emergency_special": {
        "label": "Emergency / special vehicle",
        "definition": "An ambulance, fire truck, police car, school bus, or unusual vehicle (tractor, horse carriage, oversized load).",
        "queries": ["ambulance or fire truck with lights", "police car on the road", "school bus stopped",
                    "unusual or oversized vehicle on the road"]},
    "road_anomaly": {
        "label": "Road anomaly / odd object",
        "definition": "Debris, animals, cones, roadworks, flooding or any unusual object on or beside the roadway.",
        "queries": ["debris or object lying on the road", "animal crossing the road", "traffic cones and roadworks",
                    "flooded or damaged road surface"]},
    "near_miss": {
        "label": "Near miss / close call",
        "definition": "A collision with a pedestrian, cyclist or vehicle was narrowly avoided, or a gap closed dangerously fast.",
        "queries": ["car braking hard to avoid a collision", "pedestrian very close to the moving vehicle",
                    "gap to the vehicle ahead closes fast", "two vehicles nearly collide"]},
}
