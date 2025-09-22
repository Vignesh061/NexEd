from flask import Flask, render_template, request, jsonify
import json
import os
import fitz
import numpy as np
from werkzeug.utils import secure_filename
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import time
from datetime import datetime
from difflib import SequenceMatcher

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Initialize the sentence transformer model for ATS analysis
try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("SentenceTransformer model loaded successfully")
except Exception as e:
    print(f"Error loading SentenceTransformer model: {e}")
    model = None

class CollegeSearchEngine:
    def __init__(self):
        self.colleges_data = []
        self.districts = []
        self.load_data()
    
    def load_data(self):
        """Load college data from JSON or CSV file"""
        try:
            # Try to load JSON first
            if os.path.exists('data/colleges_data.json'):
                with open('data/colleges_data.json', 'r', encoding='utf-8') as file:
                    self.colleges_data = json.load(file)
                print(f"Loaded JSON data: {len(self.colleges_data)} colleges")
            elif os.path.exists('colleges_data.json'):
                with open('colleges_data.json', 'r', encoding='utf-8') as file:
                    self.colleges_data = json.load(file)
                print(f"Loaded JSON data: {len(self.colleges_data)} colleges")
            else:
                print("No data file found. Please add colleges_data.json to the root folder or data folder.")
                # Create sample data for testing
                self.colleges_data = [
                    {
                        "Aishe Code": "C-6575",
                        "Name": "Regional Medical Research Institute (I.C.M.R.)",
                        "State": "Andaman and Nicobar Islands",
                        "District": "Nicobars",
                        "Website": "http://www.rmrc.res.in",
                        "Year Of Establishment": "1983",
                        "Location": "Urban",
                        "College Type": "Affiliated College",
                        "Management": "Central Government",
                        "University Aishe Code": "U-0369",
                        "University Name": "Pondicherry University, Puducherry",
                        "University Type": "Central University"
                    }
                ]
                return
            
            # Extract unique districts
            self.districts = list(set([college.get('District', '').strip() 
                                     for college in self.colleges_data if college.get('District')]))
            
            print(f"Loaded {len(self.colleges_data)} colleges from {len(self.districts)} districts")
            print(f"Available districts: {self.districts[:10]}...")  # Show first 10 districts
            
        except Exception as e:
            print(f"Error loading data: {e}")
            self.colleges_data = []
    
    def fuzzy_match_district(self, query, threshold=0.4):
        """Use fuzzy matching to find the best district match"""
        if not query or not self.districts:
            return None
        
        query = query.strip().lower()
        best_match = None
        best_score = 0
        
        print(f"Searching for district: '{query}'")
        
        # First try exact match
        for district in self.districts:
            if query == district.lower():
                print(f"Exact match found: {district}")
                return district
        
        # Then try partial match
        for district in self.districts:
            district_lower = district.lower()
            if query in district_lower or district_lower in query:
                score = SequenceMatcher(None, query, district_lower).ratio()
                print(f"Partial match: {district} (score: {score})")
                if score > best_score:
                    best_score = score
                    best_match = district
        
        # Finally try fuzzy matching
        if best_score < threshold:
            for district in self.districts:
                score = SequenceMatcher(None, query, district.lower()).ratio()
                if score > best_score and score >= threshold:
                    best_score = score
                    best_match = district
                    print(f"Fuzzy match: {district} (score: {score})")
        
        print(f"Best match: {best_match} (score: {best_score})")
        return best_match if best_score >= threshold else None
    
    def search_colleges_by_district(self, district_query):
        """Search colleges by district name using fuzzy matching"""
        if not district_query:
            return self.colleges_data, None
            
        # Find the best matching district
        matched_district = self.fuzzy_match_district(district_query)
        
        if not matched_district:
            print(f"No district match found for: {district_query}")
            return [], None
        
        print(f"Matched district: {matched_district}")
        
        # Find all colleges in the matched district
        colleges = [
            college for college in self.colleges_data
            if college.get('District', '').lower() == matched_district.lower()
        ]
        
        print(f"Found {len(colleges)} colleges in {matched_district}")
        return colleges, matched_district
    
    def get_all_districts(self):
        """Get all unique districts for autocomplete"""
        return sorted(self.districts)
    
    def search_colleges_multi_field(self, query):
        """Search colleges across multiple fields"""
        if not query:
            return self.colleges_data
        
        query = query.lower().strip()
        results = []
        
        for college in self.colleges_data:
            # Search in multiple fields
            searchable_fields = [
                college.get('District', ''),
                college.get('Name', ''),
                college.get('State', ''),
                college.get('College Type', ''),
                college.get('University Name', ''),
                college.get('Management', '')
            ]
            
            # Check if query matches any field
            if any(query in field.lower() for field in searchable_fields if field):
                results.append(college)
        
        return results

# ATS Analysis Functions
def extract_text_from_pdf(pdf_path):
    try:
        text = ""
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        return ""

def compute_similarity(resume_text, job_desc):
    if not model:
        return 0.0
    try:
        embeddings = model.encode([resume_text, job_desc])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])
        # Convert numpy float to Python float
        score = float(similarity[0][0] * 100)
        return round(score, 2)
    except Exception as e:
        print(f"Error computing similarity: {str(e)}")
        return 0.0

def extract_keywords_from_job(job_desc):
    # This is a simplified keyword extraction
    # In a real application, you'd use NLP techniques
    common_job_keywords = [
        "team", "develop", "manage", "lead", "project", "collaborate", 
        "analyze", "implement", "communication", "problem-solving", 
        "innovation", "customer", "solution", "strategy", "agile",
        "javascript", "python", "java", "c++", "react", "node", "database",
        "cloud", "aws", "azure", "leadership", "design", "ux"
    ]
    
    # Find keywords that appear in the job description
    found_keywords = [kw for kw in common_job_keywords if kw.lower() in job_desc.lower()]
    return found_keywords

def check_ats_friendly(resume_text, job_desc):
    issues = []
    
    # Check for complex formatting elements
    if re.search(r'\.(png|jpg|jpeg|gif)', resume_text, re.IGNORECASE):
        issues.append("Avoid using images or graphics in your resume.")
    
    # Check for common section headers
    expected_sections = ["education", "experience", "skills", "projects", "contact"]
    missing_sections = [section for section in expected_sections if section.lower() not in resume_text.lower()]
    if missing_sections:
        issues.append(f"Missing important sections: {', '.join(missing_sections)}. Add clear section headers.")
    
    # Check for readable fonts (Assume text-only PDF passes this)
    if not resume_text:
        issues.append("Resume text could not be extracted. Ensure your resume uses standard fonts.")
    
    # Check for keyword usage from job description
    job_keywords = extract_keywords_from_job(job_desc)
    matched_keywords = [kw for kw in job_keywords if kw.lower() in resume_text.lower()]
    missing_keywords = [kw for kw in job_keywords if kw.lower() not in resume_text.lower()]
    
    if len(matched_keywords) < len(job_keywords) * 0.5:
        issues.append("Your resume doesn't include enough keywords from the job description.")
    
    # Check resume length by word count
    word_count = len(resume_text.split())
    if word_count < 300:
        issues.append("Your resume seems too short. Aim for 400-800 words for optimal ATS parsing.")
    elif word_count > 1200:
        issues.append("Your resume is very long. Consider condensing to 1-2 pages for better ATS results.")
    
    return {
        "issues": issues if issues else ["Your resume appears to be ATS friendly!"],
        "missing_keywords": missing_keywords,
        "matched_keywords": matched_keywords,
        "job_keywords": job_keywords
    }

# Initialize the search engine
search_engine = CollegeSearchEngine()

# Routes
@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/signin')
def signin():
    """ Signin Page"""
    return render_template('signin.html')

@app.route('/colleges')
def colleges_page():
    """College directory page"""
    return render_template('colleges.html')

@app.route('/quiz')
def quiz_page():
    """ Quiz Page"""
    return render_template('quiz.html')

@app.route('/career')
def carrer_page():
    """ Carrer Page"""
    return render_template('career.html')

@app.route("/onboarding")
def onboarding():
    """Onboarding Page"""
    return render_template("onboarding.html")

@app.route("/profile")
def profile():
    """Profile Page"""
    return render_template("profile.html")

@app.route("/exam")
def exam():
    """Exam Page"""
    return render_template('exam.html')

@app.route("/chatagent")
def chatagent():
    """Chatagent Page"""
    return render_template('chatagent.html')
@app.route("/alumni")
def alumni():
    """ Alumni Page"""
    return render_template('alumni.html')


    
@app.route("/atsresume", methods=["GET", "POST"])
def atsresume():
    """ATS Resume Analysis Page"""
    if request.method == "POST":
        try:
            # Check if request is JSON (for AJAX) or form data
            if request.is_json:
                data = request.get_json()
                job_desc = data.get('job_description', '')
                
                # Handle file upload for JSON requests
                if 'resume_text' in data:
                    resume_text = data['resume_text']
                else:
                    return jsonify({"success": False, "error": "No resume text provided"}), 400
            else:
                # Handle form submission
                if "resume" not in request.files:
                    return render_template("atsresume.html", error="No file uploaded"), 400
                
                file = request.files["resume"]
                job_desc = request.form.get("job_desc", "")
                
                if file.filename == "" or job_desc.strip() == "":
                    return render_template("atsresume.html", error="Please provide both a resume file and job description"), 400
                
                # Check if the file is a PDF
                if not file.filename.lower().endswith('.pdf'):
                    return render_template("atsresume.html", error="Only PDF files are accepted"), 400
                    
                # Create a unique filename to avoid conflicts
                unique_filename = f"{int(time.time())}_{secure_filename(file.filename)}"
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
                file.save(filepath)
                
                try:
                    resume_text = extract_text_from_pdf(filepath)
                    if not resume_text.strip():
                        return render_template("atsresume.html", error="Could not extract text from the PDF. Make sure it contains text content."), 400
                finally:
                    # Clean up the uploaded file
                    try:
                        os.remove(filepath)
                    except:
                        pass
            
            # Perform ATS analysis
            score = compute_similarity(resume_text, job_desc)
            ats_result = check_ats_friendly(resume_text, job_desc)
            
            analysis_results = {
                "score": score,
                "issues": ats_result["issues"],
                "missing_keywords": ats_result["missing_keywords"],
                "matched_keywords": ats_result["matched_keywords"],
                "job_keywords": ats_result["job_keywords"],
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Return JSON response for AJAX requests
            if request.is_json:
                return jsonify({
                    "success": True,
                    "data": analysis_results
                })
            
            # Return rendered template for form submissions
            return render_template(
                "atsresume.html", 
                score=score, 
                ats_issues=ats_result["issues"],
                missing_keywords=ats_result["missing_keywords"],
                matched_keywords=ats_result["matched_keywords"],
                job_keywords=ats_result["job_keywords"],
                job_description=job_desc,
                analysis_date=analysis_results["analysis_date"]
            )
            
        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            if request.is_json:
                return jsonify({"success": False, "error": error_msg}), 500
            return render_template("atsresume.html", error=error_msg), 500
    
    # GET request - show the form
    return render_template('atsresume.html')

# API Routes for college search
@app.route('/api/search', methods=['POST'])
def search_colleges():
    """API endpoint for college search"""
    try: 
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No data provided'}), 400
            
        query = data.get('query', '').strip()
        search_type = data.get('type', 'district')
        
        print(f"Search request - Query: '{query}', Type: {search_type}")
        
        if search_type == 'district' and query:
            colleges, matched_district = search_engine.search_colleges_by_district(query)
            return jsonify({
                'success': True,
                'colleges': colleges,
                'matched_district': matched_district,
                'total': len(colleges)
            })
        elif search_type == 'multi':
            colleges = search_engine.search_colleges_multi_field(query)
            return jsonify({
                'success': True,
                'colleges': colleges,
                'total': len(colleges)
            })
        else:  # all or empty query
            return jsonify({
                'success': True,
                'colleges': search_engine.colleges_data,
                'total': len(search_engine.colleges_data)
            })
    
    except Exception as e:
        print(f"Search error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/districts')
def get_districts():
    """Get all districts for autocomplete"""
    try:
        districts = search_engine.get_all_districts()
        return jsonify({
            'success': True,
            'districts': districts
        })
    except Exception as e:
        print(f"Districts error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/college/<aishe_code>')
def get_college_details(aishe_code):
    """Get detailed information for a specific college"""
    try:
        college = next((c for c in search_engine.colleges_data 
                       if c.get('Aishe Code') == aishe_code), None)
        
        if college:
            return jsonify({
                'success': True,
                'college': college
            })
        else:
            return jsonify({
                'success': False,
                'error': 'College not found'
            }), 404
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# API endpoint for ATS analysis (for AJAX calls)
@app.route('/api/analyze-resume', methods=['POST'])
def analyze_resume_api():
    """API endpoint for resume analysis"""
    try:
        if "resume" not in request.files:
            return jsonify({"success": False, "error": "No file uploaded"}), 400
        
        file = request.files["resume"]
        job_desc = request.form.get("job_desc", "")
        
        if file.filename == "" or job_desc.strip() == "":
            return jsonify({"success": False, "error": "Please provide both a resume file and job description"}), 400
        
        # Check if the file is a PDF
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({"success": False, "error": "Only PDF files are accepted"}), 400
            
        # Create a unique filename to avoid conflicts
        unique_filename = f"{int(time.time())}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], unique_filename)
        file.save(filepath)
        
        try:
            resume_text = extract_text_from_pdf(filepath)
            if not resume_text.strip():
                return jsonify({"success": False, "error": "Could not extract text from the PDF"}), 400
                
            score = compute_similarity(resume_text, job_desc)
            ats_result = check_ats_friendly(resume_text, job_desc)
            
            return jsonify({
                "success": True,
                "data": {
                    "score": score,
                    "issues": ats_result["issues"],
                    "missing_keywords": ats_result["missing_keywords"],
                    "matched_keywords": ats_result["matched_keywords"],
                    "job_keywords": ats_result["job_keywords"],
                    "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            })
            
        finally:
            # Clean up the uploaded file
            try:
                os.remove(filepath)
            except:
                pass
                
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    # Create required directories
    os.makedirs('templates', exist_ok=True)
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    
    print("Starting Flask application...")
    print(f"Upload folder: {app.config['UPLOAD_FOLDER']}")
    print(f"SentenceTransformer model loaded: {model is not None}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)