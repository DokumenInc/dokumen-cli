# Release Notes

Feature flag rollout now requires an explicit owner field before a flag can be
enabled in production.

Application teams must add the owning team slug to each production flag before
their next rollout. Existing flags continue to run, but new production rollouts
will fail validation if the owner field is missing.
