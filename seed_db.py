import sys
import os
import random
from datetime import datetime, timezone, timedelta

# Add the project root to sys.path so we can import api
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api.service import TwinService
from data.schema import DecisionRecord, Domain, DOMAIN_OPTIONS

def generate_user_data(user_id: str = "demo_user", count: int = 300, mode: str = "predictable"):
    now = datetime.now(timezone.utc)
    records = []

    # Generate sequential data starting from `count` steps in the past,
    # adding random intervals of a few hours to simulate sequential decisions.
    current_time = now - timedelta(days=count / 5.0)  # Roughly 5 decisions a day

    for i in range(count):
        # Advance time by 2-6 hours randomly
        current_time += timedelta(hours=random.uniform(2, 6))
        
        domain_enum = random.choice(list(Domain))
        domain = domain_enum.value
        
        from data.schema import time_of_day, day_type
        tod = time_of_day(current_time).value
        dt = day_type(current_time).value
        
        # Determine persona-based context
        is_weekday = (dt == "weekday")
        
        if is_weekday:
            if tod == "morning":
                # Rushing to class
                location = "college"
                weather = random.choice(["clear", "cloudy"])
                if domain == "focus": state = random.choice(["light_work", "admin"])
                elif domain == "task": state = "meeting"  # Classes/Labs
                else: state = random.choice(["coffee", "snack"])
            elif tod == "afternoon":
                # Labs and lectures
                location = "college"
                weather = random.choice(["clear", "cloudy", "rain"])
                if domain == "focus": state = random.choice(["flow_state", "pomodoro"]) # Coding labs
                elif domain == "task": state = random.choice(["meeting", "deep_work"])
                else: state = random.choice(["lunch", "snack"])
            elif tod == "evening":
                # Hanging out, clubs, start of self-study
                location = "hostel"
                weather = "clear"
                if domain == "focus": state = random.choice(["light_work", "admin"])
                elif domain == "task": state = random.choice(["break", "email"]) # Club activities
                else: state = "snack"
            else: # night
                # Late night coding / assignments / gaming
                location = "hostel"
                weather = "clear"
                if domain == "focus": state = random.choice(["flow_state", "pomodoro"])
                elif domain == "task": state = "deep_work"
                else: state = random.choice(["none", "coffee"]) # Maggi or coffee
        else:
            # Weekend
            location = "hostel"
            weather = random.choice(["clear", "cloudy", "rain"])
            if tod == "morning" or tod == "afternoon":
                # Sleeping in, chilling
                if domain == "focus": state = "light_work"
                elif domain == "task": state = "break"
                else: state = random.choice(["lunch", "snack"])
            else: # evening/night
                # Catching up on assignments / coding
                if domain == "focus": state = random.choice(["flow_state", "pomodoro"])
                elif domain == "task": state = random.choice(["deep_work", "break"])
                else: state = random.choice(["none", "coffee"])

        # Context alignment: Mood and stress
        if state in ["flow_state", "deep_work"]:
            mood = random.uniform(0.7, 1.0)
            stress = "medium"
        elif state in ["break", "lunch", "snack"]:
            mood = random.uniform(0.6, 0.9)
            stress = "low"
        elif state in ["meeting", "admin", "email"]:
            mood = random.uniform(0.3, 0.6)
            stress = random.choice(["medium", "high"])
        else: # coffee, light_work, none, pomodoro
            mood = random.uniform(0.5, 0.8)
            stress = random.choice(["low", "medium"])
            
        record = DecisionRecord(
            timestamp=current_time,
            domain=domain,
            decision_made=state,
            location=location,
            weather=weather,
            time_of_day=tod,
            day_type=dt,
            mood_energy=round(mood, 2),
            stress_level=stress,
            outcome="success",
            user_id=user_id,
            source_mode="synthetic"
        )
        records.append(record)

    return records

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Seed synthetic decision data.")
    parser.add_argument("--user", type=str, default="demo_user", help="User ID to seed data for")
    parser.add_argument("--count", type=int, default=300, help="Number of records to generate")
    args = parser.parse_args()
    
    from api.service import TwinService
    service = TwinService()
    records = generate_user_data(args.user, args.count)
    service.store.append(records)
    print(f"✅ Successfully seeded {args.count} highly realistic B.Tech student data points for {args.user}!")

    # Force a retrain so the ML models generate weights and the dashboard shows accuracy
    try:
        service.retrain_models(args.user)
        print("🤖 Retrained ML models successfully.")
    except Exception as e:
        print(f"❌ Error during retrain: {e}")
