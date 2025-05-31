
import requests
import unittest
import os
import sys
import time
from PIL import Image
import io
import random

class AIImageTaxonomyAPITester(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Get the backend URL from frontend .env file
        with open('/app/frontend/.env', 'r') as f:
            for line in f:
                if line.startswith('REACT_APP_BACKEND_URL='):
                    self.base_url = line.strip().split('=')[1].strip('"\'')
                    break
        
        print(f"Using backend URL: {self.base_url}")
        self.test_image_path = self._create_test_image()
    
    def _create_test_image(self):
        """Create a test image for upload testing"""
        img_path = '/tmp/test_image.jpg'
        
        # Create a simple colored image
        img = Image.new('RGB', (300, 200), color=(
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        ))
        
        # Add some shapes to make it more interesting for AI analysis
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        
        # Draw a rectangle
        draw.rectangle(
            [(50, 50), (250, 150)],
            fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        )
        
        # Draw a circle
        draw.ellipse(
            [(100, 50), (200, 150)],
            fill=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        )
        
        img.save(img_path)
        print(f"Created test image at {img_path}")
        return img_path
    
    def test_1_health_endpoint(self):
        """Test the health check endpoint"""
        print("\n🔍 Testing health endpoint...")
        response = requests.get(f"{self.base_url}/api/health")
        
        self.assertEqual(response.status_code, 200, "Health endpoint should return 200")
        data = response.json()
        self.assertEqual(data["status"], "healthy", "Health status should be 'healthy'")
        print("✅ Health endpoint test passed")
    
    def test_2_upload_images(self):
        """Test image upload and analysis"""
        print("\n🔍 Testing image upload and analysis...")
        
        with open(self.test_image_path, 'rb') as img_file:
            files = {'files': ('test_image.jpg', img_file, 'image/jpeg')}
            response = requests.post(f"{self.base_url}/api/upload-images", files=files)
        
        self.assertEqual(response.status_code, 200, "Upload endpoint should return 200")
        data = response.json()
        
        # Check response structure
        self.assertIn("session_id", data, "Response should contain session_id")
        self.assertIn("taxonomy", data, "Response should contain taxonomy")
        self.assertIn("individual_analyses", data, "Response should contain individual_analyses")
        
        # Store session ID for next test
        self.session_id = data["session_id"]
        print(f"✅ Upload test passed - Session ID: {self.session_id}")
        
        # Check taxonomy structure
        taxonomy = data["taxonomy"]
        self.assertIn("deduplicated_hashtags", taxonomy, "Taxonomy should contain deduplicated_hashtags")
        self.assertIn("hashtag_count", taxonomy, "Taxonomy should contain hashtag_count")
        
        # Check if hashtags were generated
        hashtag_categories = taxonomy["deduplicated_hashtags"]
        total_hashtags = sum(len(hashtags) for hashtags in hashtag_categories.values())
        print(f"Generated {total_hashtags} hashtags across {len(hashtag_categories)} categories")
        self.assertGreater(total_hashtags, 0, "Should generate at least some hashtags")
        
        return self.session_id
    
    def test_3_get_taxonomy(self):
        """Test retrieving taxonomy for a specific session"""
        if not hasattr(self, 'session_id'):
            self.session_id = self.test_2_upload_images()
        
        print(f"\n🔍 Testing taxonomy retrieval for session {self.session_id}...")
        response = requests.get(f"{self.base_url}/api/taxonomy/{self.session_id}")
        
        self.assertEqual(response.status_code, 200, "Taxonomy endpoint should return 200")
        data = response.json()
        
        self.assertIn("taxonomy", data, "Response should contain taxonomy")
        self.assertEqual(data["session_id"], self.session_id, "Session ID should match")
        print("✅ Taxonomy retrieval test passed")
    
    def test_4_get_sessions(self):
        """Test retrieving all analysis sessions"""
        print("\n🔍 Testing sessions retrieval...")
        response = requests.get(f"{self.base_url}/api/sessions")
        
        self.assertEqual(response.status_code, 200, "Sessions endpoint should return 200")
        data = response.json()
        
        self.assertIn("sessions", data, "Response should contain sessions list")
        self.assertIsInstance(data["sessions"], list, "Sessions should be a list")
        
        # Check if our session is in the list
        if hasattr(self, 'session_id'):
            session_ids = [session["session_id"] for session in data["sessions"]]
            self.assertIn(self.session_id, session_ids, "Our session should be in the list")
        
        print(f"✅ Sessions retrieval test passed - Found {len(data['sessions'])} sessions")
    
    def test_5_error_handling(self):
        """Test error handling for invalid requests"""
        print("\n🔍 Testing error handling...")
        
        # Test invalid session ID
        invalid_id = "invalid-session-id"
        response = requests.get(f"{self.base_url}/api/taxonomy/{invalid_id}")
        self.assertEqual(response.status_code, 404, "Invalid session should return 404")
        print("✅ Invalid session test passed")
        
        # Test invalid file upload (empty request)
        response = requests.post(f"{self.base_url}/api/upload-images")
        self.assertNotEqual(response.status_code, 200, "Empty upload should not return 200")
        print("✅ Invalid upload test passed")

def run_tests():
    # Create a test suite
    suite = unittest.TestSuite()
    
    # Add tests in order
    test_case = AIImageTaxonomyAPITester()
    suite.addTest(AIImageTaxonomyAPITester('test_1_health_endpoint'))
    suite.addTest(AIImageTaxonomyAPITester('test_2_upload_images'))
    suite.addTest(AIImageTaxonomyAPITester('test_3_get_taxonomy'))
    suite.addTest(AIImageTaxonomyAPITester('test_4_get_sessions'))
    suite.addTest(AIImageTaxonomyAPITester('test_5_error_handling'))
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success/failure
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    sys.exit(run_tests())
