# Unified-Structure-Variable-Selection-USVS-
This repository provides python coding implementing Unified Structure Variable Selection (USVS) for respecting various selection rules

AIM:
1. methods performance comparison
2. show the benefits of specifing correct selection rule
3. as an example, show USVS can respect the selection rule that existing method cannot

# group.py #
AIM 1
Selection rule: Correlated variable selected collectively.
Compare to the group Lasso


# IfthenRule.py #
AIM 1
Selection rule: if the interaction is selected, then the main terms must be selected.
Compare to the overlapping group Lasso

# UnitRule.py #
AIM 2, 3
Selection rule: select 2 out of 4 correlated variables.
Compare to Lasso (no selection rule)

