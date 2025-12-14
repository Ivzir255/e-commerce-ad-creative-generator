
# src/data/data_ingestion.py
import pandas as pd
import boto3
from typing import List, Dict
import logging
from pathlib import Path
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataIngestion:
    def __init__(self, s3_bucket: str = None, local_path: str = None):
        self.s3_bucket = s3_bucket
        self.local_path = local_path
        self.s3_client = boto3.client('s3') if s3_bucket else None
    
    def fetch_from_csv(self, file_path: str) -> pd.DataFrame:
        """Load product data from CSV"""
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {len(df)} products")
        return df
    
    def fetch_from_s3(self, s3_key: str) -> pd.DataFrame:
        """Download and load data from S3"""
        local_file = f"/tmp/{Path(s3_key).name}"
        logger.info(f"Downloading from s3://{self.s3_bucket}/{s3_key}")
        self.s3_client.download_file(self.s3_bucket, s3_key, local_file)
        return self.fetch_from_csv(local_file)
    
    def validate_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean product data"""
        required_cols = ['product_id', 'product_name', 'category', 'description']
        
        # Check required columns
        missing = set(required_cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['product_id'])
        
        # Remove nulls in critical fields
        df = df.dropna(subset=['product_name', 'description'])
        
        # Clean text
        df['product_name'] = df['product_name'].str.strip()
        df['description'] = df['description'].str.strip()
        
        logger.info(f"Validated {len(df)} products")
        return df
    
    def save_to_feature_store(self, df: pd.DataFrame, output_path: str):
        """Save processed data to feature store"""
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved to feature store: {output_path}")
        
        # Upload to S3 if configured
        if self.s3_bucket:
            s3_key = f"feature-store/{Path(output_path).name}"
            self.s3_client.upload_file(output_path, self.s3_bucket, s3_key)
            logger.info(f"Uploaded to s3://{self.s3_bucket}/{s3_key}")
    
    def create_training_dataset(self, df: pd.DataFrame) -> List[Dict]:
        """Convert to training format"""
        training_data = []
        
        for _, row in df.iterrows():
            # Create input text
            input_text = f"Generate ad for: {row['product_name']}. Category: {row['category']}. Description: {row['description']}"
            
            # Create target (example template - in production, use real ad copies)
            target_text = self._generate_template_ad(row)
            
            training_data.append({
                'input': input_text,
                'target': target_text,
                'product_id': row['product_id']
            })
        
        return training_data
    
    def _generate_template_ad(self, row: pd.Series) -> str:
        """Generate template ad copy (placeholder for actual training data)"""
        templates = [
            f"🚀 Discover {row['product_name']}! {row['description'][:100]}. Shop now!",
            f"✨ New arrival: {row['product_name']}. {row['description'][:100]}. Limited stock!",
            f"💎 Premium {row['category']}: {row['product_name']}. {row['description'][:100]}. Order today!"
        ]
        return templates[hash(row['product_id']) % len(templates)]

if __name__ == "__main__":
    # Example usage
    ingestion = DataIngestion()
    sample_csv = "data/raw/products.csv"

    if not Path(sample_csv).exists():
        print("📝 Creating sample product data...")
        Path(sample_csv).parent.mkdir(parents=True, exist_ok=True)
        
        sample_products = pd.DataFrame([
            {
                'product_id': 'P001',
                'product_name': 'Wireless Bluetooth Headphones',
                'category': 'Electronics',
                'description': 'Premium noise-cancelling wireless headphones with 30-hour battery life and crystal-clear audio quality'
            },
            {
                'product_id': 'P002',
                'product_name': 'Organic Cotton T-Shirt',
                'category': 'Clothing',
                'description': 'Comfortable 100% organic cotton t-shirt available in multiple colors, eco-friendly and sustainable'
            },
            {
                'product_id': 'P003',
                'product_name': 'Smart LED Bulb',
                'category': 'Home',
                'description': 'WiFi-enabled color-changing LED bulb with voice control and energy-saving features'
            },
            {
                'product_id': 'P004',
                'product_name': 'Yoga Mat Premium',
                'category': 'Sports',
                'description': 'Non-slip extra thick yoga mat with carrying strap, perfect for all types of workouts'
            },
            {
                'product_id': 'P005',
                'product_name': 'Coffee Maker Automatic',
                'category': 'Kitchen',
                'description': 'Programmable coffee maker with thermal carafe, makes perfect coffee every time'
            }
        ])
        sample_products.to_csv(sample_csv, index=False)
        print(f"✅ Created sample data at {sample_csv}")


    df = ingestion.fetch_from_csv(sample_csv)
    df = ingestion.validate_data(df)
    ingestion.save_to_feature_store(df, "data/processed/products.parquet")
    
    training_data = ingestion.create_training_dataset(df)
    with open("data/processed/training_data.json", "w") as f:
        json.dump(training_data, f, indent=2)


# """
# Data Ingestion Module - Loads and prepares product data for training

# Key Concepts:
# - Validates CSV data
# - Creates training pairs (input -> target)
# - Saves to JSON for model training
# """

# import pandas as pd
# import json
# from pathlib import Path

# class DataIngestion:
#     def __init__(self):
#         self.data_dir = Path("data")
#         self.data_dir.mkdir(exist_ok=True)
    
#     def load_csv(self, filepath: str) -> pd.DataFrame:
#         """Load product data from CSV"""
#         print(f"📂 Loading data from {filepath}")
#         df = pd.read_csv(filepath)
#         print(f"✅ Loaded {len(df)} products")
#         return df
    
#     def validate(self, df: pd.DataFrame) -> pd.DataFrame:
#         """Validate required columns exist"""
#         required = ['product_id', 'product_name', 'category', 'description']
#         missing = set(required) - set(df.columns)
        
#         if missing:
#             raise ValueError(f"❌ Missing columns: {missing}")
        
#         # Clean data
#         df = df.dropna(subset=required)
#         df = df.drop_duplicates(subset=['product_id'])
        
#         print(f"✅ Validated {len(df)} products")
#         return df
    
#     def create_training_data(self, df: pd.DataFrame) -> list:
#         """Convert products to training format"""
#         training_data = []
        
#         for _, row in df.iterrows():
#             # Input: Product details
#             input_text = (
#                 f"Generate ad for: {row['product_name']}. "
#                 f"Category: {row['category']}. "
#                 f"Description: {row['description']}"
#             )
            
#             # Target: Template ad (in real use, you'd have real ad copies)
#             target_text = self._create_ad_template(row)
            
#             training_data.append({
#                 'input': input_text,
#                 'target': target_text,
#                 'product_id': row['product_id']
#             })
        
#         print(f"✅ Created {len(training_data)} training examples")
#         return training_data
    
#     def _create_ad_template(self, row: pd.Series) -> str:
#         """Generate ad template (placeholder for real training data)"""
#         templates = [
#             f"🚀 Discover {row['product_name']}! {row['description'][:80]}... Shop now!",
#             f"✨ New: {row['product_name']}. {row['description'][:80]}... Order today!",
#             f"💎 Premium {row['category']}: {row['product_name']}. Limited stock!"
#         ]
#         return templates[hash(row['product_id']) % len(templates)]
    
#     def save_training_data(self, data: list, output_path: str):
#         """Save training data as JSON"""
#         Path(output_path).parent.mkdir(parents=True, exist_ok=True)
#         with open(output_path, 'w') as f:
#             json.dump(data, f, indent=2)
#         print(f"💾 Saved training data to {output_path}")

# def main():
#     """Test the data ingestion pipeline"""
#     # Create sample data if it doesn't exist
#     sample_csv = "data/raw/products.csv"
#     if not Path(sample_csv).exists():
#         print("📝 Creating sample product data...")
#         Path(sample_csv).parent.mkdir(parents=True, exist_ok=True)
        
#         sample_products = pd.DataFrame([
#             {
#                 'product_id': 'P001',
#                 'product_name': 'Wireless Bluetooth Headphones',
#                 'category': 'Electronics',
#                 'description': 'Premium noise-cancelling wireless headphones with 30-hour battery life and crystal-clear audio quality'
#             },
#             {
#                 'product_id': 'P002',
#                 'product_name': 'Organic Cotton T-Shirt',
#                 'category': 'Clothing',
#                 'description': 'Comfortable 100% organic cotton t-shirt available in multiple colors, eco-friendly and sustainable'
#             },
#             {
#                 'product_id': 'P003',
#                 'product_name': 'Smart LED Bulb',
#                 'category': 'Home',
#                 'description': 'WiFi-enabled color-changing LED bulb with voice control and energy-saving features'
#             },
#             {
#                 'product_id': 'P004',
#                 'product_name': 'Yoga Mat Premium',
#                 'category': 'Sports',
#                 'description': 'Non-slip extra thick yoga mat with carrying strap, perfect for all types of workouts'
#             },
#             {
#                 'product_id': 'P005',
#                 'product_name': 'Coffee Maker Automatic',
#                 'category': 'Kitchen',
#                 'description': 'Programmable coffee maker with thermal carafe, makes perfect coffee every time'
#             }
#         ])
#         sample_products.to_csv(sample_csv, index=False)
#         print(f"✅ Created sample data at {sample_csv}")
    
#     # Run pipeline
#     ingestion = DataIngestion()
#     df = ingestion.load_csv(sample_csv)
#     df = ingestion.validate(df)
#     training_data = ingestion.create_training_data(df)
#     ingestion.save_training_data(training_data, "data/processed/training_data.json")
    
#     print("\n" + "="*60)
#     print("✅ DATA INGESTION COMPLETE")
#     print("="*60)
#     print(f"📊 Products processed: {len(df)}")
#     print(f"📝 Training examples: {len(training_data)}")
#     print(f"📁 Output: data/processed/training_data.json")
#     print("\nSample training example:")
#     print(json.dumps(training_data[0], indent=2))

# if __name__ == "__main__":
#     main()