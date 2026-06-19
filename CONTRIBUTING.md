# Contributing to FedShield

## Team
- B. Siri (23R11A6255)
- M. R. Meghana (23R11A6278)  
- P. Hathiram (23R11A6281)

## Branch Strategy
- `main` — stable, tested code only
- Feature branches → PR → review → merge

## Running Tests
```bash
python -m pytest tests/ -v
```

## Adding a New Node
1. Extend `FedShieldClient` in `nodes/flower_client.py`
2. Add data split logic in `start_client()`
3. Run tests before pushing