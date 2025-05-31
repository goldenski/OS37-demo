import pandas as pd
import numpy as np
import random
from faker import Faker
from datetime import datetime, timedelta
import pyarrow.parquet as pq
import pyarrow as pa

fake = Faker('es_MX')

# Major Mexican cities and states
cities_states = [
    ("CDMX", "Ciudad de México"),
    ("Guadalajara", "Jalisco"),
    ("Monterrey", "Nuevo León"),
    ("Puebla", "Puebla"),
    ("Tijuana", "Baja California")
]

insurance_types = ["IMSS", "Private", "Uninsured"]
insurance_probs = [0.6, 0.25, 0.15]

def add_missing_values(df, cols, min_pct=0.05, max_pct=0.15):
    for col in cols:
        pct = np.random.uniform(min_pct, max_pct)
        idx = df.sample(frac=pct).index
        df.loc[idx, col] = np.nan
    return df

def add_duplicates(df, pct=0.025):
    n_dup = int(len(df) * pct)
    dups = df.sample(n=n_dup, replace=True)
    return pd.concat([df, dups], ignore_index=True)

def generate_patient_id(n):
    return [f"PAT_{str(i).zfill(4)}" for i in range(1, n+1)]

def generate_patient_names(ids):
    first_names = [fake.first_name() for _ in ids]
    return [f"Paciente_{pid[-4:]}, {fname}" for pid, fname in zip(ids, first_names)]

def random_dates(start, end, n):
    start_u = int(start.timestamp())
    end_u = int(end.timestamp())
    return [datetime.fromtimestamp(random.randint(start_u, end_u)) for _ in range(n)]

def generate_visits(patient_ids, min_visits=1, max_visits=4, year_range=(2020,2024)):
    records = []
    for pid in patient_ids:
        n_visits = random.randint(min_visits, max_visits)
        # Seasonal variation: more visits in Jan-Mar and Sep-Nov
        months = np.random.choice([1,2,3,4,5,6,7,8,9,10,11,12], n_visits, p=[.13,.13,.13,.07,.05,.03,.03,.03,.13,.13,.13,.05])
        years = np.random.choice(range(year_range[0], year_range[1]+1), n_visits)
        for m, y in zip(months, years):
            day = random.randint(1,28)
            records.append((pid, datetime(y, m, day)))
    return records

def assign_city_state(n):
    cities, states = zip(*random.choices(cities_states, k=n))
    return list(cities), list(states)

def assign_gender(n, f_ratio=0.52):
    return np.random.choice(['F','M'], size=n, p=[f_ratio, 1-f_ratio])

def assign_insurance(n):
    return np.random.choice(insurance_types, size=n, p=insurance_probs)

def diabetes_dataset():
    n_patients = 5000
    patient_ids = generate_patient_id(n_patients)
    patient_names = generate_patient_names(patient_ids)
    dob = [fake.date_of_birth(minimum_age=45, maximum_age=70) for _ in patient_ids]
    gender = assign_gender(n_patients, f_ratio=0.57)
    city, state = assign_city_state(n_patients)
    insurance = assign_insurance(n_patients)
    # Each patient has 1-4 visits
    visits = generate_visits(patient_ids)
    data = []
    for pid, visit_date in visits:
        idx = int(pid[-4:])-1
        # ICD-10 codes and diagnosis
        diagnosis_code = random.choice(["E11", "E10", "E13"])
        diagnosis_name = {"E11": "Type 2 Diabetes", "E10": "Type 1 Diabetes", "E13": "Other diabetes"}[diagnosis_code]
        medication = random.choice(["Metformin", "Glipizide", "Insulin Glargine", "Insulin Aspart", "GLP-1 agonist"])
        medication_cost = round(random.uniform(10, 120),2)
        outcome_score = round(np.clip(random.normalvariate(7, 1.5), 4, 12),2)
        hba1c = round(random.uniform(5.5, 12.0),2)
        retinopathy = np.random.choice([0,1], p=[0.85,0.15])
        neuropathy = np.random.choice([0,1], p=[0.90,0.10])
        data.append({
            "patient_id": pid,
            "patient_name": patient_names[idx],
            "dob": dob[idx],
            "gender": gender[idx],
            "city": city[idx],
            "state": state[idx],
            "visit_date": visit_date.date(),
            "diagnosis_code": diagnosis_code,
            "diagnosis_name": diagnosis_name,
            "medication": medication,
            "medication_cost_usd": medication_cost,
            "insurance_type": insurance[idx],
            "outcome_score": outcome_score,
            "hba1c": hba1c,
            "retinopathy": retinopathy,
            "neuropathy": neuropathy
        })
    df = pd.DataFrame(data)
    df = add_missing_values(df, ["medication", "medication_cost_usd", "insurance_type", "hba1c"])
    df = add_duplicates(df)
    df.reset_index(drop=True, inplace=True)
    return df

def oncology_dataset():
    n_patients = 5000
    patient_ids = generate_patient_id(n_patients)
    patient_names = generate_patient_names(patient_ids)
    dob = [fake.date_of_birth(minimum_age=35, maximum_age=75) for _ in patient_ids]
    gender = assign_gender(n_patients, f_ratio=0.55)
    city, state = assign_city_state(n_patients)
    insurance = assign_insurance(n_patients)
    cancer_types = np.random.choice(["Breast", "Lung", "Colorectal", "Other"], p=[0.3,0.25,0.2,0.25], size=n_patients)
    icd_map = {"Breast": ("C50", "Breast cancer"), "Lung": ("C34", "Lung cancer"), "Colorectal": ("C18", "Colorectal cancer"), "Other": ("C80", "Other cancer")}
    visits = generate_visits(patient_ids)
    data = []
    for pid, visit_date in visits:
        idx = int(pid[-4:])-1
        cancer = cancer_types[idx]
        diagnosis_code, diagnosis_name = icd_map[cancer]
        medication = random.choice(["Chemotherapy", "Targeted Therapy", "Immunotherapy"])
        medication_cost = round(random.uniform(200, 2000),2)
        outcome_score = round(np.clip(random.normalvariate(30, 15), 0, 60),2)
        tnm = f"T{random.randint(1,4)}N{random.randint(0,3)}M{random.randint(0,1)}"
        survival_months = random.randint(0,60)
        data.append({
            "patient_id": pid,
            "patient_name": patient_names[idx],
            "dob": dob[idx],
            "gender": gender[idx],
            "city": city[idx],
            "state": state[idx],
            "visit_date": visit_date.date(),
            "diagnosis_code": diagnosis_code,
            "diagnosis_name": diagnosis_name,
            "medication": medication,
            "medication_cost_usd": medication_cost,
            "insurance_type": insurance[idx],
            "outcome_score": outcome_score,
            "cancer_type": cancer,
            "tnm_stage": tnm,
            "survival_months": survival_months
        })
    df = pd.DataFrame(data)
    df = add_missing_values(df, ["medication", "medication_cost_usd", "insurance_type", "tnm_stage", "survival_months"])
    df = add_duplicates(df)
    df.reset_index(drop=True, inplace=True)
    return df

def cardiology_dataset():
    n_patients = 5000
    patient_ids = generate_patient_id(n_patients)
    patient_names = generate_patient_names(patient_ids)
    dob = [fake.date_of_birth(minimum_age=40, maximum_age=80) for _ in patient_ids]
    gender = assign_gender(n_patients, f_ratio=0.49)
    city, state = assign_city_state(n_patients)
    insurance = assign_insurance(n_patients)
    condition_types = np.random.choice(["Hypertension", "CAD", "Heart Failure"], p=[0.4,0.3,0.3], size=n_patients)
    icd_map = {"Hypertension": ("I10", "Hypertension"), "CAD": ("I25", "Coronary Artery Disease"), "Heart Failure": ("I50", "Heart Failure")}
    visits = generate_visits(patient_ids)
    data = []
    for pid, visit_date in visits:
        idx = int(pid[-4:])-1
        cond = condition_types[idx]
        diagnosis_code, diagnosis_name = icd_map[cond]
        medication = random.choice(["ACE inhibitor", "Beta-blocker", "Statin"])
        medication_cost = round(random.uniform(15, 200),2)
        outcome_score = round(np.clip(random.normalvariate(60, 20), 10, 100),2)
        systolic = random.randint(100, 180)
        diastolic = random.randint(60, 110)
        bp = f"{systolic}/{diastolic}"
        cardiac_event = np.random.choice([0,1], p=[0.97,0.03])
        data.append({
            "patient_id": pid,
            "patient_name": patient_names[idx],
            "dob": dob[idx],
            "gender": gender[idx],
            "city": city[idx],
            "state": state[idx],
            "visit_date": visit_date.date(),
            "diagnosis_code": diagnosis_code,
            "diagnosis_name": diagnosis_name,
            "medication": medication,
            "medication_cost_usd": medication_cost,
            "insurance_type": insurance[idx],
            "outcome_score": outcome_score,
            "condition": cond,
            "blood_pressure": bp,
            "cardiac_event": cardiac_event
        })
    df = pd.DataFrame(data)
    df = add_missing_values(df, ["medication", "medication_cost_usd", "insurance_type", "blood_pressure"])
    df = add_duplicates(df)
    df.reset_index(drop=True, inplace=True)
    return df

def main():
    print("Generating Diabetes Mexico dataset...")
    diabetes = diabetes_dataset()
    diabetes.to_parquet("diabetes_mexico.parquet", index=False)
    print("Saved diabetes_mexico.parquet")

    print("Generating Oncology Mexico dataset...")
    oncology = oncology_dataset()
    oncology.to_parquet("oncology_mexico.parquet", index=False)
    print("Saved oncology_mexico.parquet")

    print("Generating Cardiology Mexico dataset...")
    cardiology = cardiology_dataset()
    cardiology.to_parquet("cardiology_mexico.parquet", index=False)
    print("Saved cardiology_mexico.parquet")

if __name__ == "__main__":
    main()
