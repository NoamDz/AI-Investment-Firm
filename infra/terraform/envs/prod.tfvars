# Prod environment overrides (Plan 4 T31).
#
# Bigger sizes than dev because prod runs the full 30-ticker universe with
# real research traffic — burstable t4g + 1 vCPU cannot sustain steady-state
# load. db.r6g.large gives consistent memory-optimised perf for Postgres;
# 2 vCPU / 4 GB matches the orchestrator + agents resident set under load.
env               = "prod"
db_instance_class = "db.r6g.large"
ecs_task_cpu      = 2048
ecs_task_memory   = 4096
