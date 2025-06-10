#!/usr/bin/env python3
"""
Synthetic CMS 1500 Claims Data Generator

This script generates synthetic healthcare claims data in CMS 1500 format
using publicly available CPT, HCPCS, and ICD-10 codes.

Author: AI Assistant
Date: 2025
"""

import random
import csv
import json
from datetime import datetime, timedelta
from faker import Faker
import uuid

# Initialize Faker for generating realistic data
fake = Faker()

class CMS1500Generator:
    def __init__(self):
        self.setup_code_tables()
        self.setup_provider_data()
        
    def setup_code_tables(self):
        """Initialize medical code tables with common codes"""
        
        # Common CPT Codes (Current Procedural Terminology)
        self.cpt_codes = {
            '99213': {'description': 'Office visit, established patient, moderate complexity', 'price_range': (150, 250)},
            '99214': {'description': 'Office visit, established patient, high complexity', 'price_range': (200, 350)},
            '99203': {'description': 'Office visit, new patient, moderate complexity', 'price_range': (250, 400)},
            '99204': {'description': 'Office visit, new patient, high complexity', 'price_range': (350, 500)},
            '85025': {'description': 'Complete blood count with differential', 'price_range': (25, 45)},
            '80053': {'description': 'Comprehensive metabolic panel', 'price_range': (30, 60)},
            '93000': {'description': 'Electrocardiogram', 'price_range': (50, 100)},
            '71020': {'description': 'Chest X-ray, 2 views', 'price_range': (80, 150)},
            '73721': {'description': 'MRI lower extremity without contrast', 'price_range': (1200, 2000)},
            '29881': {'description': 'Arthroscopy, knee, surgical', 'price_range': (3000, 5000)},
            '45378': {'description': 'Colonoscopy, diagnostic', 'price_range': (800, 1500)},
            '36415': {'description': 'Venipuncture', 'price_range': (15, 30)},
            '90471': {'description': 'Immunization administration', 'price_range': (25, 40)},
            '12001': {'description': 'Simple repair of superficial wounds', 'price_range': (100, 200)},
            '81001': {'description': 'Urinalysis', 'price_range': (20, 35)}
        }
        
        # Common HCPCS Codes (Healthcare Common Procedure Coding System)
        self.hcpcs_codes = {
            'J3420': {'description': 'Injection, vitamin B-12', 'price_range': (20, 40)},
            'A4253': {'description': 'Blood glucose test strips', 'price_range': (30, 60)},
            'E0784': {'description': 'External ambulatory infusion pump', 'price_range': (200, 400)},
            'L3806': {'description': 'Knee orthosis', 'price_range': (150, 300)},
            'G0439': {'description': 'Annual wellness visit', 'price_range': (150, 250)},
            'Q4081': {'description': 'Injection, epoetin alfa', 'price_range': (100, 200)}
        }
        
        # Common ICD-10 Diagnosis Codes
        self.icd10_codes = {
            'Z00.00': 'Encounter for general adult medical examination without abnormal findings',
            'I10': 'Essential hypertension',
            'E11.9': 'Type 2 diabetes mellitus without complications',
            'M79.89': 'Other specified soft tissue disorders',
            'J06.9': 'Acute upper respiratory infection, unspecified',
            'K21.9': 'Gastro-esophageal reflux disease without esophagitis',
            'M25.561': 'Pain in right knee',
            'R06.02': 'Shortness of breath',
            'R50.9': 'Fever, unspecified',
            'N39.0': 'Urinary tract infection, site not specified',
            'F32.9': 'Major depressive disorder, single episode, unspecified',
            'M54.5': 'Low back pain',
            'H52.4': 'Presbyopia',
            'L70.0': 'Acne vulgaris',
            'R51': 'Headache',
            'Z23': 'Encounter for immunization',
            'S61.001A': 'Unspecified open wound of right thumb without damage to nail, initial encounter',
            'K59.00': 'Constipation, unspecified',
            'R05': 'Cough',
            'E78.5': 'Hyperlipidemia, unspecified'
        }
        
        # Place of Service Codes
        self.pos_codes = {
            '11': 'Office',
            '12': 'Home',
            '21': 'Inpatient Hospital',
            '22': 'Outpatient Hospital',
            '23': 'Emergency Room - Hospital',
            '81': 'Independent Laboratory',
            '99': 'Other Place of Service'
        }
        
        # Type of Service Codes
        self.tos_codes = {
            '1': 'Medical care',
            '2': 'Surgery',
            '3': 'Consultation',
            '4': 'Diagnostic X-ray',
            '5': 'Diagnostic laboratory',
            '6': 'Radiation therapy',
            '7': 'Anesthesia',
            '8': 'Assistant at surgery'
        }
        
    def setup_provider_data(self):
        """Initialize provider and facility data"""
        self.providers = [
            {'name': 'Dr. Sarah Johnson', 'npi': '1234567890', 'specialty': 'Family Practice', 'tax_id': '12-3456789'},
            {'name': 'Dr. Michael Chen', 'npi': '2345678901', 'specialty': 'Internal Medicine', 'tax_id': '23-4567890'},
            {'name': 'Dr. Emily Rodriguez', 'npi': '3456789012', 'specialty': 'Cardiology', 'tax_id': '34-5678901'},
            {'name': 'Dr. David Wilson', 'npi': '4567890123', 'specialty': 'Orthopedics', 'tax_id': '45-6789012'},
            {'name': 'Dr. Lisa Thompson', 'npi': '5678901234', 'specialty': 'Dermatology', 'tax_id': '56-7890123'}
        ]
        
        self.facilities = [
            {'name': 'City Medical Center', 'address': '123 Main St', 'city': 'Springfield', 'state': 'IL', 'zip': '62701'},
            {'name': 'Regional Health Clinic', 'address': '456 Oak Ave', 'city': 'Chicago', 'state': 'IL', 'zip': '60601'},
            {'name': 'Suburban Family Practice', 'address': '789 Elm Dr', 'city': 'Naperville', 'state': 'IL', 'zip': '60540'},
            {'name': 'Metro Specialty Center', 'address': '321 Pine St', 'city': 'Peoria', 'state': 'IL', 'zip': '61602'},
            {'name': 'Community Health Partners', 'address': '654 Cedar Ln', 'city': 'Rockford', 'state': 'IL', 'zip': '61101'}
        ]
        
        self.insurance_plans = [
            {'name': 'Blue Cross Blue Shield', 'payer_id': 'BCBS001', 'group_number': 'GRP12345'},
            {'name': 'Aetna', 'payer_id': 'AETNA01', 'group_number': 'GRP23456'},
            {'name': 'UnitedHealthcare', 'payer_id': 'UHC0001', 'group_number': 'GRP34567'},
            {'name': 'Cigna', 'payer_id': 'CIGNA01', 'group_number': 'GRP45678'},
            {'name': 'Humana', 'payer_id': 'HUMANA1', 'group_number': 'GRP56789'},
            {'name': 'Medicare', 'payer_id': 'MEDICARE', 'group_number': 'MEDICARE'},
            {'name': 'Medicaid', 'payer_id': 'MEDICAID', 'group_number': 'MEDICAID'}
        ]
    
    def generate_patient_data(self):
        """Generate synthetic patient information"""
        gender = random.choice(['M', 'F'])
        birth_date = fake.date_of_birth(minimum_age=18, maximum_age=85)
        
        patient = {
            'patient_id': fake.unique.random_number(digits=8),
            'first_name': fake.first_name_male() if gender == 'M' else fake.first_name_female(),
            'last_name': fake.last_name(),
            'middle_initial': fake.random_letter().upper(),
            'date_of_birth': birth_date.strftime('%m/%d/%Y'),
            'gender': gender,
            'address': fake.street_address(),
            'city': fake.city(),
            'state': fake.state_abbr(),
            'zip_code': fake.zipcode(),
            'phone': fake.phone_number(),
            'ssn': fake.ssn(),
            'member_id': fake.random_number(digits=10, fix_len=True)
        }
        return patient
    
    def generate_claim_data(self):
        """Generate a complete CMS 1500 claim"""
        patient = self.generate_patient_data()
        provider = random.choice(self.providers)
        facility = random.choice(self.facilities)
        insurance = random.choice(self.insurance_plans)
        
        # Generate service date (within last 90 days)
        service_date = fake.date_between(start_date='-90d', end_date='today')
        
        # Select diagnosis codes (1-4 diagnoses)
        num_diagnoses = random.randint(1, 4)
        diagnosis_codes = random.sample(list(self.icd10_codes.keys()), num_diagnoses)
        
        # Generate procedure lines (1-6 lines)
        num_procedures = random.randint(1, 6)
        procedure_lines = []
        
        total_charges = 0
        
        for i in range(num_procedures):
            # Randomly choose between CPT and HCPCS codes
            if random.random() < 0.8:  # 80% CPT, 20% HCPCS
                code_type = 'CPT'
                code = random.choice(list(self.cpt_codes.keys()))
                code_info = self.cpt_codes[code]
            else:
                code_type = 'HCPCS'
                code = random.choice(list(self.hcpcs_codes.keys()))
                code_info = self.hcpcs_codes[code]
            
            # Generate charges within the code's price range
            min_price, max_price = code_info['price_range']
            charges = round(random.uniform(min_price, max_price), 2)
            total_charges += charges
            
            procedure_line = {
                'line_number': i + 1,
                'service_date': service_date.strftime('%m/%d/%Y'),
                'place_of_service': random.choice(list(self.pos_codes.keys())),
                'procedure_code': code,
                'procedure_description': code_info['description'],
                'modifier': random.choice(['', '25', '59', 'RT', 'LT']) if random.random() < 0.3 else '',
                'diagnosis_pointer': ','.join([str(j+1) for j in range(min(len(diagnosis_codes), random.randint(1, 2)))]),
                'charges': charges,
                'units': random.randint(1, 3),
                'rendering_provider_npi': provider['npi']
            }
            procedure_lines.append(procedure_line)
        
        # Generate claim header information
        claim = {
            'claim_id': str(uuid.uuid4()),
            'claim_number': fake.random_number(digits=12, fix_len=True),
            'patient_control_number': fake.random_number(digits=8, fix_len=True),
            'type_of_service': random.choice(list(self.tos_codes.keys())),
            'place_of_service': procedure_lines[0]['place_of_service'],  # Use first line's POS
            'submission_date': datetime.now().strftime('%m/%d/%Y'),
            'onset_date': service_date.strftime('%m/%d/%Y') if random.random() < 0.3 else '',
            'total_charges': round(total_charges, 2),
            'amount_paid': round(total_charges * random.uniform(0.7, 0.95), 2),  # Simulate partial payment
            'patient_signature_on_file': 'Y',
            'assignment_of_benefits': random.choice(['Y', 'N']),
            'prior_authorization': fake.random_number(digits=10) if random.random() < 0.2 else '',
            
            # Patient information
            'patient': patient,
            
            # Insurance information
            'primary_insurance': {
                'insurance_name': insurance['name'],
                'payer_id': insurance['payer_id'],
                'group_number': insurance['group_number'],
                'policy_number': fake.random_number(digits=12, fix_len=True)
            },
            
            # Provider information
            'billing_provider': {
                'name': provider['name'],
                'npi': provider['npi'],
                'tax_id': provider['tax_id'],
                'specialty': provider['specialty']
            },
            
            # Facility information
            'service_facility': facility,
            
            # Diagnosis information
            'diagnoses': [
                {
                    'code': code,
                    'description': self.icd10_codes[code],
                    'pointer': i + 1
                }
                for i, code in enumerate(diagnosis_codes)
            ],
            
            # Procedure lines
            'procedure_lines': procedure_lines
        }
        
        return claim
    
    def generate_multiple_claims(self, num_claims=100):
        """Generate multiple claims with memory optimization for large datasets"""
        claims = []
        batch_size = 10000  # Process in batches to manage memory
        
        print(f"Generating {num_claims:,} claims...")
        
        for i in range(num_claims):
            claim = self.generate_claim_data()
            claims.append(claim)
            
            # Progress indicator
            if (i + 1) % 1000 == 0:
                print(f"Generated {i + 1:,} claims...")
            
            # Memory management for very large datasets
            if len(claims) >= batch_size and num_claims > batch_size:
                # For very large datasets, consider yielding batches instead of storing all in memory
                pass
        
        return claims
    
    def generate_claims_streaming(self, num_claims, batch_size=10000):
        """Generator function for memory-efficient large-scale claim generation"""
        print(f"Streaming generation of {num_claims:,} claims in batches of {batch_size:,}...")
        
        for i in range(0, num_claims, batch_size):
            current_batch_size = min(batch_size, num_claims - i)
            batch = []
            
            for j in range(current_batch_size):
                claim = self.generate_claim_data()
                batch.append(claim)
                
                if (i + j + 1) % 1000 == 0:
                    print(f"Generated {i + j + 1:,} claims...")
            
            yield batch
    
    def export_to_csv_streaming(self, num_claims, filename='cms1500_claims.csv', batch_size=10000):
        """Export claims to CSV with streaming for large datasets"""
        fieldnames = [
            'claim_id', 'claim_number', 'patient_control_number', 'submission_date',
            'patient_id', 'patient_first_name', 'patient_last_name', 'patient_dob',
            'patient_gender', 'patient_address', 'patient_city', 'patient_state', 'patient_zip',
            'insurance_name', 'policy_number', 'group_number',
            'provider_name', 'provider_npi', 'provider_specialty',
            'service_date', 'place_of_service', 'total_charges', 'amount_paid',
            'primary_diagnosis', 'secondary_diagnosis', 'procedure_codes', 'procedure_descriptions'
        ]
        
        print(f"Exporting {num_claims:,} claims to {filename} using streaming method...")
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            total_written = 0
            for batch in self.generate_claims_streaming(num_claims, batch_size):
                for claim in batch:
                    # Flatten the nested data structure for CSV
                    row = {
                        'claim_id': claim['claim_id'],
                        'claim_number': claim['claim_number'],
                        'patient_control_number': claim['patient_control_number'],
                        'submission_date': claim['submission_date'],
                        'patient_id': claim['patient']['patient_id'],
                        'patient_first_name': claim['patient']['first_name'],
                        'patient_last_name': claim['patient']['last_name'],
                        'patient_dob': claim['patient']['date_of_birth'],
                        'patient_gender': claim['patient']['gender'],
                        'patient_address': claim['patient']['address'],
                        'patient_city': claim['patient']['city'],
                        'patient_state': claim['patient']['state'],
                        'patient_zip': claim['patient']['zip_code'],
                        'insurance_name': claim['primary_insurance']['insurance_name'],
                        'policy_number': claim['primary_insurance']['policy_number'],
                        'group_number': claim['primary_insurance']['group_number'],
                        'provider_name': claim['billing_provider']['name'],
                        'provider_npi': claim['billing_provider']['npi'],
                        'provider_specialty': claim['billing_provider']['specialty'],
                        'service_date': claim['procedure_lines'][0]['service_date'],
                        'place_of_service': claim['place_of_service'],
                        'total_charges': claim['total_charges'],
                        'amount_paid': claim['amount_paid'],
                        'primary_diagnosis': claim['diagnoses'][0]['code'] if claim['diagnoses'] else '',
                        'secondary_diagnosis': claim['diagnoses'][1]['code'] if len(claim['diagnoses']) > 1 else '',
                        'procedure_codes': ';'.join([line['procedure_code'] for line in claim['procedure_lines']]),
                        'procedure_descriptions': ';'.join([line['procedure_description'] for line in claim['procedure_lines']])
                    }
                    writer.writerow(row)
                    total_written += 1
                
                # Flush to disk periodically
                csvfile.flush()
                print(f"Written {total_written:,} claims to CSV...")
        
        print(f"Export complete! {total_written:,} claims written to {filename}")
    
    def export_to_csv_line_items(self, num_claims, filename='cms1500_line_items.csv', batch_size=10000):
        """Export individual procedure lines (for 1M+ line items) with streaming"""
        fieldnames = [
            'claim_id', 'claim_number', 'line_number', 'patient_control_number', 'submission_date',
            'patient_id', 'patient_first_name', 'patient_last_name', 'patient_dob', 'patient_gender',
            'patient_address', 'patient_city', 'patient_state', 'patient_zip',
            'insurance_name', 'policy_number', 'group_number',
            'provider_name', 'provider_npi', 'provider_specialty',
            'service_date', 'place_of_service', 'procedure_code', 'procedure_description',
            'modifier', 'diagnosis_pointer', 'charges', 'units', 'rendering_provider_npi',
            'diagnosis_1', 'diagnosis_2', 'diagnosis_3', 'diagnosis_4'
        ]
        
        print(f"Exporting line items from {num_claims:,} claims to {filename}...")
        
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            total_lines = 0
            for batch in self.generate_claims_streaming(num_claims, batch_size):
                for claim in batch:
                    # Create a row for each procedure line
                    for line in claim['procedure_lines']:
                        row = {
                            'claim_id': claim['claim_id'],
                            'claim_number': claim['claim_number'],
                            'line_number': line['line_number'],
                            'patient_control_number': claim['patient_control_number'],
                            'submission_date': claim['submission_date'],
                            'patient_id': claim['patient']['patient_id'],
                            'patient_first_name': claim['patient']['first_name'],
                            'patient_last_name': claim['patient']['last_name'],
                            'patient_dob': claim['patient']['date_of_birth'],
                            'patient_gender': claim['patient']['gender'],
                            'patient_address': claim['patient']['address'],
                            'patient_city': claim['patient']['city'],
                            'patient_state': claim['patient']['state'],
                            'patient_zip': claim['patient']['zip_code'],
                            'insurance_name': claim['primary_insurance']['insurance_name'],
                            'policy_number': claim['primary_insurance']['policy_number'],
                            'group_number': claim['primary_insurance']['group_number'],
                            'provider_name': claim['billing_provider']['name'],
                            'provider_npi': claim['billing_provider']['npi'],
                            'provider_specialty': claim['billing_provider']['specialty'],
                            'service_date': line['service_date'],
                            'place_of_service': line['place_of_service'],
                            'procedure_code': line['procedure_code'],
                            'procedure_description': line['procedure_description'],
                            'modifier': line['modifier'],
                            'diagnosis_pointer': line['diagnosis_pointer'],
                            'charges': line['charges'],
                            'units': line['units'],
                            'rendering_provider_npi': line['rendering_provider_npi'],
                            'diagnosis_1': claim['diagnoses'][0]['code'] if len(claim['diagnoses']) > 0 else '',
                            'diagnosis_2': claim['diagnoses'][1]['code'] if len(claim['diagnoses']) > 1 else '',
                            'diagnosis_3': claim['diagnoses'][2]['code'] if len(claim['diagnoses']) > 2 else '',
                            'diagnosis_4': claim['diagnoses'][3]['code'] if len(claim['diagnoses']) > 3 else ''
                        }
                        writer.writerow(row)
                        total_lines += 1
                
                # Flush to disk and show progress
                csvfile.flush()
                if total_lines % 10000 == 0:
                    print(f"Written {total_lines:,} line items...")
        
        print(f"Export complete! {total_lines:,} line items written to {filename}")
        return total_lines
    
    def export_to_json(self, claims, filename='cms1500_claims.json'):
        """Export claims to JSON format"""
        with open(filename, 'w', encoding='utf-8') as jsonfile:
            json.dump(claims, jsonfile, indent=2, ensure_ascii=False)
        print(f"Claims exported to {filename}")
    
    def print_claim_summary(self, claim):
        """Print a formatted summary of a single claim"""
        print(f"\n{'='*60}")
        print(f"CLAIM SUMMARY - {claim['claim_number']}")
        print(f"{'='*60}")
        print(f"Patient: {claim['patient']['first_name']} {claim['patient']['last_name']}")
        print(f"DOB: {claim['patient']['date_of_birth']}")
        print(f"Provider: {claim['billing_provider']['name']}")
        print(f"Service Date: {claim['procedure_lines'][0]['service_date']}")
        print(f"Total Charges: ${claim['total_charges']:.2f}")
        print(f"\nDiagnoses:")
        for dx in claim['diagnoses']:
            print(f"  {dx['code']}: {dx['description']}")
        print(f"\nProcedures:")
        for line in claim['procedure_lines']:
            print(f"  {line['procedure_code']}: {line['procedure_description']} - ${line['charges']:.2f}")

def main():
    """Main function to demonstrate the CMS 1500 generator"""
    print("CMS 1500 Synthetic Claims Data Generator")
    print("=" * 50)
    
    # Initialize the generator
    generator = CMS1500Generator()
    
    # Get user input for number of claims
    num_claims_input = input("Enter number of claims to generate (default 10): ") or "10"
    
    try:
        num_claims = int(num_claims_input)
    except ValueError:
        print("Invalid input. Using default of 10 claims.")
        num_claims = 10
    
    # For large datasets, use streaming methods
    if num_claims > 50000:
        print(f"\nLarge dataset detected ({num_claims:,} claims).")
        print("Using memory-efficient streaming generation...")
        
        # Export format choice
        print("\nExport options for large datasets:")
        print("1. CSV (claim-level data)")
        print("2. CSV (line-item level data) - Creates more rows but detailed")
        print("3. Both formats")
        
        choice = input("Choose export format (1/2/3): ") or "1"
        
        if choice in ['1', '3']:
            filename = f'cms1500_claims_{num_claims}.csv'
            generator.export_to_csv_streaming(num_claims, filename)
        
        if choice in ['2', '3']:
            filename = f'cms1500_line_items_{num_claims}.csv'
            total_lines = generator.export_to_csv_line_items(num_claims, filename)
            print(f"Total line items created: {total_lines:,}")
        
        print(f"\nLarge dataset generation complete!")
        
    else:
        # For smaller datasets, use in-memory generation
        print(f"\nGenerating {num_claims:,} synthetic claims...")
        claims = generator.generate_multiple_claims(num_claims)
        
        # Show a sample claim
        if claims:
            print("\nSample Claim:")
            generator.print_claim_summary(claims[0])
        
        # Export options
        export_choice = input("\nExport format (csv/json/both/none): ").lower()
        
        if export_choice in ['csv', 'both']:
            generator.export_to_csv(claims)
        
        if export_choice in ['json', 'both']:
            generator.export_to_json(claims)
        
        print(f"\nGeneration complete! Created {len(claims):,} synthetic claims.")
    
    print("\nPerformance Notes:")
    print("- For 1M+ claims, expect 30-60 minutes generation time")
    print("- Line-item export creates ~3-4 rows per claim (more detailed)")
    print("- Memory usage is optimized for large datasets")
    print("- CSV files will be large (1M claims ≈ 500MB-1GB)")
    print("\nNote: This data is completely synthetic and should only be used for testing purposes.")


def benchmark_performance():
    """Benchmark function to test generation speed"""
    print("Performance Benchmark")
    print("=" * 30)
    
    generator = CMS1500Generator()
    
    # Test with smaller batches first
    test_sizes = [100, 1000, 10000]
    
    for size in test_sizes:
        print(f"\nTesting {size:,} claims...")
        start_time = datetime.now()
        
        claims = generator.generate_multiple_claims(size)
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"Generated {len(claims):,} claims in {duration:.2f} seconds")
        print(f"Rate: {len(claims)/duration:.0f} claims/second")
        
        # Estimate time for 1M claims
        estimated_time_1m = (1000000 / (len(claims)/duration)) / 60
        print(f"Estimated time for 1M claims: {estimated_time_1m:.1f} minutes")
    
    print("\nBenchmark complete!")


if __name__ == "__main__":
    # Install required packages if not already installed
    try:
        from faker import Faker
    except ImportError:
        print("Installing required package: faker")
        import subprocess
        subprocess.check_call(["pip", "install", "faker"])
        from faker import Faker
    
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--benchmark':
        benchmark_performance()
    else:
        main()
