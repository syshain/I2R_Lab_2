# Lab 2 — Worksheet Answers & Hints

Companion document for `worksheet.docx`. Questions that are **theoretical** (answerable from first principles) get a full worked answer. Questions that depend on **experimental observation** (running the script, collecting data, reading the printed output) get a *skeletal hint* describing what to look for and how to reason about it — the actual numbers come from your run.

Answers are keyed to the task/question text in the worksheet.

---

## Part A — Numerical Inverse Kinematics

### Joint Limits table (fill-in)

Full theoretical answer — these are fixed by the robot datasheet, not an experiment. From the UFACTORY Lite 6 user manual, Section 4 "Joint Range":

| Joint | Min (°) | Max (°) |
|-------|---------|---------|
| J1    | −360    | +360    |
| J2    | −150    | +150    |
| J3    | −3.5    | +300    |
| J4    | −360    | +360    |
| J5    | −124    | +124    |
| J6    | −360    | +360    |

The asymmetric one is **J3**, whose minimum is −3.5° rather than a symmetric ± value. (These match the `joint_limits_deg` block already at the top of `ik_numerical.m`.)

---

### Task A0 — Implement Forward Kinematics

**(a) Fill-in for `fk_lite6.m`.** The Standard DH transform for joint *i*, with θᵢ = qᵢ + θ_offset,i, is:

```
Ti = [ cos(theta), -sin(theta)*cos(alpha(i)),  sin(theta)*sin(alpha(i)), a(i)*cos(theta);
       sin(theta),  cos(theta)*cos(alpha(i)), -cos(theta)*sin(alpha(i)), a(i)*sin(theta);
       0,           sin(alpha(i)),              cos(alpha(i)),            d(i);
       0,           0,                          0,                         1 ]
```

So the blanks are:

- `theta = q(i) + theta_offset(i)`
- Row 1: `[ cos(theta), -sin(theta)*cos(alpha(i)), sin(theta)*sin(alpha(i)), a(i)*cos(theta) ]`
- Row 2: `[ sin(theta), cos(theta)*cos(alpha(i)), -cos(theta)*sin(alpha(i)), a(i)*sin(theta) ]`
- Row 3: `[ 0, sin(alpha(i)), cos(alpha(i)), d(i) ]`
- Row 4: `[ 0, 0, 0, 1 ]`

Two unit traps to watch: in the provided skeleton the DH table stores `d` and `a` in **mm** and angles in **degrees**, so convert with `d = dh(:,2)/1000`, `a = dh(:,4)/1000`, `alpha = deg2rad(dh(:,3))`, `theta_offset = deg2rad(dh(:,1))` before building the matrices. (If you forget the /1000, positions come out ~1000× too large.)

**(b) Sanity check — all joints zero.** This is partly experimental (run it), but the *expected* result is predictable, so here's the reasoning. With q = 0 every θᵢ equals its offset, which for this DH table folds the middle links back toward the base rather than stretching them out. Chaining the six transforms puts the end-effector at roughly **z ≈ +133 mm** (≈ d₁ + d₄ = 243.3 + 227.6, less the small wrist offset) with only a couple of millimetres of horizontal offset — i.e. the EE hangs almost directly above the base, well below the ~530 mm max reach. That low, near-vertical result is exactly what you should see; if instead your EE lands far out horizontally or at a very different height, a DH entry (most commonly a sign on one of the α angles, e.g. +90 entered where −90 belongs) is wrong.

*Skeletal hint:* run `T_zero = fk_lite6(zeros(6,1)); disp(T_zero(1:3,4)*1000)` and confirm it agrees with the controller-reported pose when you physically home the robot. Compare both the height and the horizontal offset, not just the total distance — a mirrored/side-swapped placement often preserves distance but flips the x/y sign, which is how α sign errors reveal themselves.

---

### Task A1 — Run the Demonstration Script

Both sub-questions are **experimental** — they depend on your specific target poses and initial guesses.

*Skeletal hints:*

- **How many iterations does each test require?** Watch the `iter-detailed` output from `lsqnonlin`/`fsolve` (set via `'Display','iter-detailed'` in `ik_solve.m`). Read off the final iteration index per test. Expect the easy cases (Test 1, near home) to converge in a handful of iterations, and the harder initialisations to take more. If a test hits the cap (`MaxIterations = 500`) without settling, note that separately — it signals a poor start, not a broken solver.
- **What is the final error norm?** It's the `error_norm` returned by `ik_solve` (the residual norm). On success it should be tiny — well below the `FunctionTolerance`/`StepTolerance` thresholds (1e-10 / 1e-12), i.e. effectively machine-precision. Record the magnitude; if it's O(1e-8) or smaller the solve succeeded, if it plateaus at something like 1e-2 the target wasn't reachable from that guess.

---

### Task A2 — Analyze Solution Multiplicity

**Question:** Group the converged solutions. How many distinct solutions did you find? Does this match your expectation for a 6-DOF arm?

This is **experimental** (you must actually run ≥4 different initial guesses), but the *expectation* part is theoretical, so here's the framework to judge your result against.

Theory: for many 6-DOF arms the IK problem admits multiple solutions because of discrete symmetries — classically elbow up/down, wrist flipped/unflipped, and shoulder left/right, which for an arm with a spherical wrist combine into up to 2 × 2 × 2 = 8 branches. Whether *this* Lite 6 reaches that maximum depends on its exact wrist geometry, so treat "several" (typically 2–4 easily recovered from different seeds) as the realistic expectation rather than a hard guarantee of 8. Crucially, not every branch is always feasible — some fall outside the joint limits or beyond the workspace — so the count you can actually recover is at most the theoretical maximum, and often less.

*Skeletal hint:* after collecting your converged `q_sol` vectors, cluster them by rounding to a coarse grid (e.g. round each angle to the nearest 10°, then unique-row the matrix). Count the clusters. Ask yourself: do the members of each cluster differ mainly in one joint (elbow = J3 region) or in the wrist trio (J4/J5/J6)? That tells you which symmetry produced the branch. If you only found 1–2 branches, it usually means your initial guesses were clustered in the same basin — spread them further apart (try a clearly elbow-down seed vs an elbow-up seed) before concluding the arm has fewer solutions.

---

### Task A3 — Explore Failure Modes

**Question:** Solve IK for a target far outside the workspace (x = 800 mm > max reach ~530 mm). What happens? Does the solver report failure? What is the residual error?

Mixed — the mechanism is theoretical, the exact numbers are experimental.

Theory: the target is unreachable, so there is no q with e(q) = 0. `lsqnonlin` is a least-squares solver, not a feasibility checker, so it will *not* throw an error — it converges to the **closest achievable pose**, i.e. the boundary point of the workspace nearest the target. Because the arm saturates against its joint limits while chasing an impossible point, the solution tends to land on a limit boundary (one or more joints pinned at min/max).

*Skeletal hint:* run it and inspect three things: (1) `success` — likely still `true` because the optimiser minimised the residual successfully even though the residual isn't zero; this is the trap, "success" means *optimisation converged*, not *pose achieved*. (2) the residual/error norm — expect it to plateau at a clearly non-zero value (centimetres-scale position error), unlike the ~machine-zero of Tasks A1/A2. (3) `q_sol` — check whether any joint is sitting exactly on its limit. That combination (large residual + joint(s) pinned at a bound) is the signature of an unreachable target. Contrast this residual against the ~1e-8 residuals from the reachable tests to make the distinction crisp.

---

## Part B — Vision-Assisted DH & Hand-Eye Calibration

### Task B1 — Study the Calibration Pipeline

**(1) `fk_from_zeta()` fill-in.** Same Standard DH construction as Task A0, just in Python. The blanks:

```python
theta = q[i] + th_off[i]                      # <-- fill in
...
T = T @ np.array([
    [ct,  -st*ca,  st*sa,  a[i]*ct],          # <-- row 1
    [st,   ct*ca, -ct*sa,  a[i]*st],          # <-- row 2
    [0,    sa,     ca,     d[i]],             # <-- row 3
    [0,    0,      0,      1 ],
])
```

One thing to be careful about: `fk_from_zeta` does **no unit conversion itself** — it uses whatever units `zeta` is already in. The docstring label is inconsistent, so don't rely on it; the authoritative statement of the convention is in `run_calibration_twostage.py`'s `print_dh_table()`, which prints the DH corrections with units `[m] [rad] [m] [rad]` for the a / α / d / θ blocks respectively. So when you call it for the cross-check, feed `a` and `d` in metres and the angles in radians (i.e. build ζ the same way `load_calibration_data` provides `zeta_nominal`). Cross-checking against your MATLAB `fk_lite6` at q = 0 and nominal ζ should then give the same position vector to within floating-point round-off; a mismatch by ~1000× or a rotated orientation almost always means a unit-conversion slip (mm vs m) in one of the two implementations.

**(2) `levenberg_marquardt()` — role of λ and why it adapts.** Full theoretical answer.

λ blends between two solvers. Each step solves `(JᵀJ + λI)Δx = −Jᵀr` (equivalently the SVD form `d = s/(s²+λ)` used in the code). When λ is **small**, the term is dominated by `JᵀJ` and the step is the **Gauss-Newton** step — fast quadratic convergence, but it assumes the linear model is accurate and can overshoot badly when you're far from the solution or the Jacobian is ill-conditioned. When λ is **large**, `λI` dominates and the step shrinks toward a small **gradient-descent** step — slow but robust, guaranteed-ish to reduce the cost. So λ trades speed against stability.

It's adapted because neither extreme is right everywhere: early on, when the fit is poor and the linearisation unreliable, you want a big λ (cautious steps); once you're close and the linear model is trustworthy, you want λ small (fast GN steps). The code implements exactly this policy: after each trial step, if the new cost is lower it accepts the step and **decreases** λ (by a factor 0.1–0.5 depending on how much the cost dropped — bigger drop → more aggressive reduction), trusting Gauss-Newton more; if the cost went up it rejects the step and **multiplies λ by 3**, retreating toward gradient descent until a step finally improves things. This self-tuning is what makes LM robust enough to start from a rough hand-eye initialisation without diverging.

**(3) `rotation_log_map()` / `rotation_exp_map()` — why log maps instead of subtracting rotation matrices or Euler angles?** Full theoretical answer (this is the key conceptual question of the lab).

Positions live in flat ℝ³, so a position error is just `p_ref − p_i`. Orientations do **not** live in a flat space — the set of valid rotations SO(3) is a curved 3-dimensional surface embedded in 9-dimensional matrix space. Two naive approaches both fail:

- **Subtracting rotation matrices** (`R_ref − R_i`): the difference of two orthogonal matrices is generally *not* orthogonal, so it doesn't represent any real rotation. There's no meaningful "direction" or "magnitude" of orientation error you can extract from it.
- **Subtracting Euler angles**: depends entirely on which convention (ZYX vs XYZ, intrinsic vs extrinsic) you chose, and it breaks down at **gimbal lock**, where one angle becomes undefined and two axes collapse. Near those singularities a tiny physical rotation produces a huge, discontinuous jump in the Euler values, so the "difference" is neither smooth nor proportional to the true angular separation.

Instead we do this. First combine the two orientations into a single rotation representing only their disagreement: `R_rel = R_refᵀ · R_i`. If the orientations match, `R_rel = I` (zero error); otherwise `R_rel` is "the little rotation that takes one to the other." Then `rotation_log_map(R_rel)` converts that one rotation into a plain 3-vector (axis-angle): its **length** is the geodesic angle between the two orientations (in radians) and its **direction** is the axis of rotation. That vector behaves exactly like a position error — you can scale it, add it, and feed it straight into the Jacobian — so the orientation rows flow through the same least-squares machinery as the position rows.

Why is this accurate? Near the identity the curved SO(3) surface looks locally flat, and the log map reads the error on that local flat patch (the tangent space). Our residuals stay small because we iterate toward consistency, so we never wander far from the identity and the linear approximation stays valid. Concretely, in the code `vec_from_transform()` extracts `[position, log(R)]` for each predicted board pose, and `compute_residuals_and_jacobian()` forms `r_i = V₀ − V_i`; the orientation half of that subtraction is legitimate precisely because each `V` already carries its orientation as a log-map vector, not as a raw matrix or Euler triple.

*(A compact version of this rationale is also written into `procedures.md` §B2 as a boxed aside.)*

---

### Task B2 — Collect Calibration Data

No graded questions — this is a procedure. Practical notes to maximise calibration quality (feed directly into the "how can it be improved" question in B3):

- **Pose diversity beats pose count.** 60 nearly-identical poses teach the optimizer less than 40 well-spread ones. Deliberately vary every joint, especially J2/J3 (elbow) and the wrist trio, and sweep the board across different camera viewpoints. Coverage of the *motion manifold* is what makes parameters observable.
- **Keep the board square-on-ish and well-framed.** Extreme oblique views inflate ArUco reprojection error and get poses rejected (the script discards high-reprojection-error captures). Aim for moderate viewing angles with the whole board comfortably inside the frame.
- **Rigid everything.** The base must not move and the board must be rigidly fixed; any creep during collection corrupts the "board is stationary" assumption that the whole method rests on.
- **Enough equations.** With 30 unknowns you need ≫ 30 independent constraints; ~60 poses gives 6×60 = 360 residual rows, comfortably overdetermined.

---

### Task B3 — Run the Two-Stage Calibration

All of these are **experimental** — the numbers come from your dataset. Below is a skeletal guide to interpreting each, plus the one genuinely theoretical question (how to improve).

**Stage 1 results:**

- *How much did T_ee_cam change from initialization?* Compare the printed Stage-1 T_ee_cam against the seed (from `init_ee_cam_two_pose`/Horaud). Expect a noticeable shift — Stage 1 exists precisely to absorb the bulk of the systematic hand-eye error, so a change of centimetres and/or tens of degrees in the mount is normal and desirable. If it barely moved, either the init was already good or the data lacks leverage on the hand-eye parameters.
- *Remaining T_base_board spread after HE-only?* Look at the mean position/orientation spread reported after Stage 1. It should be much smaller than the pre-calibration spread but rarely zero — the leftover is the part attributable to DH error, which Stage 1 (DH frozen) cannot remove.

**Stage 2 results:**

- *Which DH parameters received the largest corrections?* Diff the final ζ against nominal, grouped by type (all six a's, all six α's, all six d's, all six θ_offsets). Report the biggest movers.
- *Are the corrections physically plausible?* Judge against manufacturing tolerance: link-length changes (a, d) should be sub-millimetre to low-single-digit-mm; joint-offset changes (θ) should be fractions of a degree to a couple of degrees. Corrections of tens of mm or many degrees are a red flag pointing to bad data, a mis-set initialisation, or a structural-coupling artefact rather than a real mechanical offset.
- *Did Stage-2 T_ee_cam shift significantly vs Stage 1?* Small drift is expected (the two are jointly re-estimated). A large swing suggests the hand-eye and DH blocks are fighting over the same error — the L2 regularisation (`reg_weight = 1e-2` toward nominal) is there to keep this bounded.

**Comparison table:**

- *Percentage improvement of joint over HE-only?* Compute the relative drop in the T_base_board spread metric (mean position error in mm and/or mean orientation error in deg) from Stage 1 → Stage 2: `improvement% = (spread_HE − spread_joint)/spread_HE × 100`. Do it separately for position and orientation.
- *Is it worth the complexity, or is HE-only "good enough"?* Frame it against the accuracy target. If HE-only already lands within the repeatability requirement for your application, the extra 24-parameter stage buys marginal gain for added fragility. If you need sub-millimetre absolute positioning, the joint stage is usually necessary because HE-only leaves the DH-induced bias uncorrected. Let your two spread numbers decide.
- *How can the calibration be improved?* (theoretical) Draw on the data-collection notes above and the pipeline structure: increase pose diversity and total count; tighten ArUco detection (better lighting, larger/closer board, higher-resolution camera, stricter reprojection threshold); verify the board and base rigidity; widen the motion envelope to excite weakly-observable parameters (especially the coupled a/d pairs); consider weighting position vs orientation appropriately; and if a parameter is structurally unobservable (near-zero singular value), freeze it or supply a prior rather than letting it float. Reducing sensor noise lowers the floor the optimizer can reach, since LM stops at the noise level, not at zero.

---

### Task B4 — Observability Analysis

First two sub-questions are **experimental**; the third is **theoretical**.

- *Roughly how many singular values are clearly larger than the smallest few?* Run the uncommented SVD lines and read the spectrum `s`. You'll see a cluster of relatively large values tapering down to a tail of near-zero ones. The count of the large cluster approximates the number of **effectively identifiable** parameter combinations (out of 30). Expect noticeably fewer than 30 — the gap between the "bulk" and the "tail" is the observability margin. Record both the count and the condition number `s[0]/s[-1]`; a very large condition number (>1e6) confirms strong ill-conditioning.
- *Name one DH parameter with a large correction and one that barely moved; explain.* Pull from the Stage-2 diff. Typical pattern: a link length (say a₂ or d₄) moves a lot while a neighbouring twist/offset barely budges. Explain via coupling — see next item.
- *Why are some DH parameters harder to identify than others?* (theoretical) Many DH parameters enter the forward kinematics **redundantly**: moving one parameter a little can be compensated by nudging another, producing almost no change in the end-effector trajectory. Classic examples are the couplings between a link's own `d` and the adjacent link's `a`/`α`, and between a joint's `θ_offset` and the previous link's `a`. Because several different parameter vectors predict nearly identical board poses, the Jacobian columns for those parameters are near-linearly dependent, so the corresponding directions are weakly excited in the data and show up as small singular values. The optimizer can therefore only pin down certain *combinations* of the parameters, not each individually — hence the SVD mask freezes the unobservable directions and the L2 regularisation pulls them back toward nominal. Parameters tied to motions you actually exercised (big sweeps of J2/J3) tend to be better observed than those affecting axes you barely moved.

---

*Prepared as a staff/instructor companion to `worksheet.docx`. Experimental-answer items are intentionally left as interpretation guides rather than fixed numbers, since correct values depend on each group's collected data.*
