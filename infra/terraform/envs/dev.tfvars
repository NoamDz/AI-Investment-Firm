# Dev environment overrides (Plan 4 T31).
#
# Only `env` needs to be set — every other variable's default in variables.tf
# is already tuned for dev:
#   region            = "us-east-1"      (single-region, cheapest)
#   project_name      = "firm"
#   vpc_cidr          = "10.0.0.0/16"
#   db_instance_class = "db.t4g.micro"   (smallest burstable Graviton)
#   ecs_task_cpu      = 1024             (1 vCPU)
#   ecs_task_memory   = 2048             (2 GB)
env = "dev"
