## The failure mode: **tight face clearance** (internal tag `gap`)

*Written by the engineer who measured it, from a held-out evaluation of the
frozen base policy. Qualitative on purpose — the numbers are deliberately
withheld so that what you infer from the video is yours.*

The two cubes start close enough that there is very little clear space between
their nearest **faces**.

"Close" here is not the distance between their centres, and that distinction is
the whole of this failure mode. Each cube is 40 mm across, but a cube presented
corner-first spans 56.6 mm along that direction. Two cubes whose centres are a
fixed distance apart can therefore be comfortably clear of one another or almost
touching, depending only on how they happen to be turned. The environment
rejects placements closer than about 59 mm centre-to-centre, so the tightest
configurations are the ones where both cubes are turned toward each other
corner-first — the region is as much a statement about **yaw** as about
proximity.

**What the policy does.** It reaches and grasps the red cube about as reliably
as it does anywhere else. Grasping is not the problem. What collapses is getting
the red cube onto the green one: a large share of these episodes end with the
red cube grasped, or dropped, but never placed.

**What it looks like in the video.** The gripper comes down into the narrow slot
between the two cubes. Because the wrist orientation is frozen at reset, the
fingers cannot rotate to line up with the gap, so a finger or the hand body
catches the neighbouring green cube on the way down or on the way back up. The
green cube gets shoved. Sometimes it is nudged a few millimetres and the
attempted stack then lands off-target; sometimes it is knocked well clear, after
which there is nothing left to stack onto. Occasionally the disturbance happens
on the descent and the grasp itself is spoiled.

**Where the policy is already fine.** Once the red cube *is* on the green one,
these episodes hold together about as well as any other. The defect is
specifically in the approach and placement, not in the settling.

**What a fix would have to do.** Make the approach and the descent respect the
neighbouring cube: get the gripper into and out of that slot without disturbing
the thing being stacked onto.
