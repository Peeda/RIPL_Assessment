## The failure mode: **target cube at the edge of the workspace** (internal tag `farb`)

*Written by the engineer who measured it, from a held-out evaluation of the
frozen base policy. Qualitative on purpose — the numbers are deliberately
withheld so that what you infer from the video is yours.*

The green cube — the one being stacked *onto* — starts far from the robot's
base, near the outer edge of what the arm can comfortably reach. The red cube
starts somewhere comfortable, and the two cubes are well clear of each other.

Note the asymmetry, because it is what makes this a distinct mode: only the
**target** is far. Configurations where *both* cubes are at the edge are a
different and less interesting problem — there the arm's inverse kinematics
simply saturate and no amount of policy correction recovers a target it cannot
reach. Those are excluded here on purpose. In this region the arm can and does
reach everything.

**What the policy does.** It grasps the red cube essentially every time —
grasping is flawless here — and it gets the cube over to the green one and
released about as often as it does anywhere. What collapses is the stack
**staying**: the cube is placed and then does not end the episode sitting
still on top of its target.

**What it looks like in the video.** At full extension the arm is less steady.
The release happens with the cube slightly off-centre, or with residual sideways
motion, or from marginally too high; the cube lands on an edge or a corner, rocks,
and slides or topples off. Sometimes the arm's own retreat brushes it. The
placement looks nearly right in the frame before it fails, and then it does not
settle.

**Where the policy is already fine.** Reaching, grasping, and transporting are
not the problem, and the neighbouring cube is not being disturbed — there is
plenty of clearance between the two.

**What a fix would have to do.** Improve the quality of the release and the
final settle at the far edge of the workspace: place the cube squarely and with
low residual motion, and let go cleanly, so the stack is still standing when the
episode ends.
