# v3 - Refactored to modularise and cleanup
## 9 Oct 2026
### Restructured with the following major changes
- Python based ik only
- reference/ removed, not committed
- Broken orphans removed
- Only skeleton is in ik_lite6.py -> one <TO_FILL> activity
- Added observability script that can be run independently and called by other scripts
- Updated STRUCTURE.md 
- Added reporting.py to make result reporting easier and modular.
- All code compiles clean

# v2 - Canonical restructure + cleanup
16 Sep 2026
Restructured Lab 2 to the shared src/data/docs/skeletons/reference template (matches Lab 3).
- python_dh_optimization/ -> src/ ; Calibration/ -> data/ ; Docs/ -> docs/
- MATLAB_IK/ -> reference/MATLAB_IK/ (frozen Part A provenance, not on Python import path)
- VerticalArucoBoardDetector.py (broken orphan) -> reference/
- Path fixes: _CALIB_DIR -> _DATA_DIR resolving ../data in collect_data.py, run_calibration_twostage.py, aruco_board_detector.py
- Stripped verbose docstrings/banners from src/; condensed to terse form
- Generated skeletons/ (AST stubbed mirror of src/, all bodies raise NotImplementedError)
- Added STRUCTURE.md documenting layout + module roles
- Updated worksheet.docx + procedures.md path refs (MATLAB_IK/, python_dh_optimization/, Lab_2/ik_numerical.m)
Verified: py_compile clean on src/ + skeletons/; full import DAG resolves (xArm stubbed); offline calibration smoke test passes (60 poses, residual 7.65 -> 4.40).
UNCOMMITTED - pending review.

# v1 - First commit
21 Aug 2026
Working lab 2 code 
- Inverse kinematics on matlab
- Python based dh optimization and verification

