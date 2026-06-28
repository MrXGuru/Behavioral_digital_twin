import time
from api.service import TwinService
from seed_db import generate_user_data

def run():
    print("Generating data...")
    records = generate_user_data("test_user_2", 200, "predictable")
    service = TwinService()
    service.store.append(records)
    print("Retraining...")
    t0 = time.time()
    service.retrain_models("test_user_2")
    print(f"Done in {time.time()-t0:.2f}s")

if __name__ == "__main__":
    run()
