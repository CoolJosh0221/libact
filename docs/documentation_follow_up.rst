Documentation Follow-up Inventory
=================================

This infrastructure change intentionally defers substantive v1 documentation.
Follow-up work should cover:

* a versioned installation guide and a final README documentation pass;
* the stable public API declaration and deprecation/support policies;
* the capability matrix;
* batch-querying and stopping-criterion guides;
* migration guides, including guidance for modAL and libact 0.1 users;
* a contributor guide and repository-level ``CONTRIBUTING.md``;
* validation or replacement of stale overview, ALBL, cost-sensitive, example,
  and extension-development prose;
* replacement or archival of the unavailable legacy Hierarchical Sampling
  dataset link in the multiclass API reference;
* missing diagrams and maintained end-to-end examples.

The intended future destinations are the user guide for workflows and
capabilities, the API reference for the stable API policy, a migration section
for migration guides, and the development section for contribution guidance.
No placeholder page is linked until its content exists.

Hosted-site administration
--------------------------

Repository-side Read the Docs configuration is present, but activating it for
the existing ``libact`` project requires project-owner access:
``BLOCKED-ON-PI``.

The GitHub ``Documentation / Build documentation`` check is ready for branch
protection, but marking it required also needs repository administrator access:
``BLOCKED-ON-PI``.
