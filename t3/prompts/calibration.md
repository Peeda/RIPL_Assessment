## Calibrating your region against the environment's own distribution

The failure mode below is described in words on purpose: what you infer about
its *mechanism* should come from the video and the description, not from a table
of our measurements.

Its *location* is a different question, and one you can answer for yourself. A
region is a claim about where in a distribution something sits — and you have
been given that distribution in full. `_initialize_episode` is in the
environment source above, with its sampler, its bounds, its rejection radius and
the shared offset both cubes receive; `_load_agent` gives the robot base. Before
you commit to any threshold, derive from that source:

1. **The shape of the axis you are targeting.** Whatever quantity the failure
   description points at — clearance between the cubes' faces, distance from the
   base, something else — work out what the environment's own sampler makes of
   it: its typical value, and how far its tail actually reaches. This is
   arithmetic on code you have been given, not a guess.

2. **How narrow the region has to be.** The base policy is competent; a failure
   mode it hit often would not be worth training a residual against. These
   regions are thin corners of the distribution, not halves of it. If the region
   you have written down would capture a large share of nominal draws, you have
   described the distribution rather than a corner of it.

3. **The far end of the tail, not the near end.** Phrases like "at the edge of
   the workspace" or "the faces are nearly touching" name an *extreme*. A region
   that stops short of that extreme — or worse, one that sits entirely below the
   middle of the true corner — trains the residual on the easy part of the
   problem while it is scored on the hard part. When a bound is given to you as
   the limit of what the environment can produce, sample up to it. Do not leave
   yourself a safety margin inside the region you were asked to target.

State this derivation in your `rationale`: which quantity you chose, what you
worked out its nominal distribution to be, and what fraction of nominal draws
you expect your region to capture. That last figure is checked against the
sampler you actually wrote.
