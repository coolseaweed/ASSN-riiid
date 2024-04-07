# riiid_project
Riiid project


# Train
## Env setup

## Run training
```bash
docker compose --profile train up -d 
docker exec riiid-p3-train-1 python train.py [--overwrite]
```