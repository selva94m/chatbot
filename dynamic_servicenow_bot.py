import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import string
import os

# Ensure NLTK data is available
def ensure_nltk_resources():
    resources = ['punkt', 'stopwords', 'wordnet']
    for resource in resources:
        try:
            nltk.data.find(f'corpora/{resource}' if resource in ['stopwords', 'wordnet'] else f'tokenizers/{resource}')
        except LookupError:
            print(f"Downloading {resource}...")
            nltk.download(resource)

# Call this at the start to make sure we have all resources
ensure_nltk_resources()

class DynamicServiceNowBot:
    def __init__(self, data_path=None):
        """
        Initialize the ServiceNow bot with a dataset of questions and solutions.
        If no data path is provided, it will generate sample data.
        
        Args:
            data_path (str, optional): Path to the Excel file containing questions and solutions
        """
        # Initialize lemmatizer
        self.lemmatizer = WordNetLemmatizer()
        
        # Load the data
        if data_path and os.path.exists(data_path):
            self.data = pd.read_excel(data_path)
            print(f"Loaded data from {data_path}")
        else:
            print("No data file provided or file doesn't exist. Generating sample data...")
            self.data = self.generate_sample_data()
            if data_path:
                self.data.to_excel(data_path, index=False)
                print(f"Sample data saved to {data_path}")
        
        # Ensure the data has the required columns
        required_columns = ['question', 'solution']
        if not all(col in self.data.columns for col in required_columns):
            raise ValueError(f"Data must contain these columns: {required_columns}")
        
        # Extract keywords from questions
        self.data['keywords'] = self.data['question'].apply(self.extract_keywords)
        
        # Preprocess the questions
        self.data['processed_question'] = self.data['question'].apply(self.preprocess_text)
        
        # Initialize the TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        self.question_vectors = self.vectorizer.fit_transform(self.data['processed_question'])
        
        print(f"Bot initialized with {len(self.data)} question-solution pairs")
    
    def generate_sample_data(self):
        """Generate sample Azure DevOps questions and solutions"""
        data = [
            {
                "question": "Azure DevOps pipeline failing with 'Unable to connect to the remote server'",
                "solution": "This error typically occurs due to network connectivity issues. Check your agent's network connection and proxy settings. Ensure the agent can reach Azure DevOps services. If using self-hosted agents, verify the agent service is running and has internet access. Try restarting the agent or provisioning a new agent if issues persist."
            },
            {
                "question": "How do I set up branch policies in Azure DevOps?",
                "solution": "1. Navigate to your Azure DevOps project.\n2. Go to Repos > Branches.\n3. Find the branch you want to add policies to and click the three dots.\n4. Select 'Branch policies'.\n5. Configure desired policies such as requiring minimum number of reviewers, linked work items, or build validation.\n6. Save your changes.\nBranch policies will now be enforced when pull requests target this branch."
            },
            {
                "question": "Cannot access Azure DevOps - getting 'TF400813: The user is not authorized' error",
                "solution": "This error indicates a permissions issue. Verify:\n1. Your account has access to the organization/project.\n2. Your license is active.\n3. Ask an administrator to check your permissions.\n4. If using PAT (Personal Access Token), ensure it's valid and has required scopes.\n5. Try clearing browser cache or using incognito mode.\n6. If using SSO, verify your Azure AD account has proper access."
            },
            {
                "question": "How to restore a deleted Azure DevOps project?",
                "solution": "Azure DevOps projects can be restored within 28 days of deletion:\n1. Sign in to Azure DevOps as an organization administrator.\n2. Go to Organization Settings > Overview.\n3. Select 'Projects' from the sidebar.\n4. Click on 'Deleted projects' tab.\n5. Find the project you want to restore and click 'Restore'.\n6. Confirm the restoration.\nNote: If the project doesn't appear, it may have been permanently deleted or the 28-day recovery period has expired."
            },
            {
                "question": "Azure DevOps build agent offline or not connecting",
                "solution": "To troubleshoot an offline build agent:\n1. Verify the agent machine has network connectivity.\n2. Check if the agent service is running (services.msc on Windows).\n3. Examine agent logs in _diag folder for errors.\n4. Ensure the agent's PAT hasn't expired.\n5. Try restarting the agent service.\n6. If self-hosted, reconfigure the agent with './config.cmd' (Windows) or './config.sh' (Linux/macOS).\n7. Check if agent pool is at capacity or if you need additional parallel jobs."
            },
            {
                "question": "How to increase timeout for Azure DevOps pipelines?",
                "solution": "You can increase pipeline timeout in two ways:\n\n1. In YAML pipelines:\nAdd this at the top of your YAML file:\n```yaml\npool:\n  vmImage: 'ubuntu-latest'\njob:\n  timeoutInMinutes: 120 # Set to desired timeout value (default is 60)\n```\n\n2. For classic pipelines:\n- Edit the pipeline\n- Select Options\n- Under 'Job timeout' select custom and set your desired timeout\n- Save\n\nNote: Organization settings may impose maximum limits on timeouts."
            },
            {
                "question": "Getting 'No hosted parallelism has been purchased or granted' error in Azure DevOps",
                "solution": "This error occurs when you've exhausted your parallel job quota. Solutions:\n1. Wait for currently running pipelines to complete.\n2. Purchase additional Microsoft-hosted parallel jobs.\n3. Set up self-hosted agents which don't count against this limit.\n4. Check if you're using the free tier which provides 1 parallel job with 1800 minutes/month.\n5. For public projects, check if you're eligible for the free grant of parallel jobs.\nTo purchase more parallelism, go to Organization Settings > Billing > Parallel jobs."
            }
        ]
        return pd.DataFrame(data)
    
    def extract_keywords(self, text):
        """Extract important keywords from text"""
        if not isinstance(text, str):
            return []
        
        # Convert to lowercase
        text = text.lower()
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and punctuation
        stop_words = set(stopwords.words('english'))
        keywords = [word for word in tokens if word not in stop_words and word not in string.punctuation]
        
        # Include important Azure DevOps terms even if they're stopwords
        azure_terms = ['azure', 'devops', 'pipeline', 'build', 'release', 'agent', 'repo', 'git', 'project']
        keywords.extend([word for word in tokens if word in azure_terms])
        
        return list(set(keywords))  # Remove duplicates
    
    def preprocess_text(self, text):
        """
        Preprocess text by converting to lowercase, removing special characters,
        lemmatizing, and removing stopwords.
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        filtered_tokens = [word for word in tokens if word not in stop_words]
        
        # Lemmatize
        lemmatized_tokens = [self.lemmatizer.lemmatize(word) for word in filtered_tokens]
        
        # Join back into a string
        return " ".join(lemmatized_tokens)
    
    def get_solution(self, query, threshold=0.3, max_results=3):
        """
        Get the solution for a query by finding the most similar question in the dataset.
        
        Args:
            query (str): The user's query
            threshold (float): Minimum similarity score to consider a match
            max_results (int): Maximum number of results to return
            
        Returns:
            dict: Contains best solution, alternatives, and confidence scores
        """
        # Preprocess the query
        processed_query = self.preprocess_text(query)
        
        # Extract keywords from the query
        query_keywords = set(self.extract_keywords(query))
        
        # Vectorize the query
        query_vector = self.vectorizer.transform([processed_query])
        
        # Calculate similarity with all questions
        similarity_scores = cosine_similarity(query_vector, self.question_vectors).flatten()
        
        # Boost scores for questions with matching keywords
        for idx, row in self.data.iterrows():
            question_keywords = set(row['keywords'])
            # Calculate Jaccard similarity between keyword sets
            if query_keywords and question_keywords:
                keyword_similarity = len(query_keywords.intersection(question_keywords)) / len(query_keywords.union(question_keywords))
                # Boost the similarity score based on keyword matches
                similarity_scores[idx] += keyword_similarity * 0.2  # Weight of keyword matching
                
        # Sort by similarity score
        sorted_indices = np.argsort(similarity_scores)[::-1]  # Descending order
        top_indices = sorted_indices[:max_results]
        
        # Get the best match
        best_match_idx = top_indices[0]
        best_match_score = similarity_scores[best_match_idx]
        
        # No good matches found
        if best_match_score < threshold:
            return {
                "solution": "I'm sorry, I couldn't find a relevant solution for your query. Please try rephrasing or provide more details about your Azure DevOps issue.",
                "confidence": 0,
                "matched_question": None,
                "alternatives": []
            }
        
        # Get alternative matches
        alternatives = []
        for idx in top_indices[1:]:
            score = similarity_scores[idx]
            if score >= threshold:
                alternatives.append({
                    "question": self.data.loc[idx, 'question'],
                    "solution": self.data.loc[idx, 'solution'],
                    "confidence": float(score)
                })
        
        # Return the solution with confidence score
        return {
            "solution": self.data.loc[best_match_idx, 'solution'],
            "confidence": float(best_match_score),
            "matched_question": self.data.loc[best_match_idx, 'question'],
            "alternatives": alternatives
        }
    
    def interactive_session(self):
        """
        Start an interactive session with the bot.
        """
        print("Azure DevOps Support Bot is ready! Type 'exit' to quit.")
        print("You can ask questions about Azure DevOps issues and I'll try to help.")
        while True:
            query = input("\nEnter your query: ")
            if query.lower() in ['exit', 'quit', 'bye']:
                print("Goodbye!")
                break
            
            result = self.get_solution(query)
            
            if result['matched_question']:
                print(f"\nMatched question: {result['matched_question']}")
                print(f"Confidence: {result['confidence']:.2f}")
                print(f"\nSolution: {result['solution']}")
                
                # Show alternatives if available
                if result['alternatives']:
                    print("\nOther possible solutions:")
                    for i, alt in enumerate(result['alternatives'], 1):
                        print(f"\n{i}. Related to: {alt['question']}")
                        print(f"   Confidence: {alt['confidence']:.2f}")
            else:
                print(f"\n{result['solution']}")


# Run the bot
if __name__ == "__main__":
    bot = DynamicServiceNowBot("azure_devops_servicenow_sample.xlsx")
    bot.interactive_session()
