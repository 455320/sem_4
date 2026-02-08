# Prepare a Python module to analyze the strength of a given password.
"""
Input:
    password : str

Output (Dict):
    strength : str
    score : int
    feedback : str
"""

class PasswordStrengthAnalyzer:

    MIN_LENGTH = 12
    MAX_LENGTH = 64
    
    def check(self, password: str) -> dict:
      
        if not isinstance(password, str):
            return {
                "strength": "INVALID",
                "score": 0,
                "feedback": "Input must be a string."
            }
            
        password = password.strip()
        
        if not password:
            return {
                "strength": "INVALID",
                "score": 0,
                "feedback": "Password cannot be empty."
            }

        length = len(password)
        
        # Character type counters
        lower_chars = sum(1 for c in password if c.islower())
        upper_chars = sum(1 for c in password if c.isupper())
        numbers = sum(1 for c in password if c.isdigit())
        special_chars = sum(1 for c in password if not c.isalnum())
        
        has_lower = lower_chars > 0
        has_upper = upper_chars > 0
        has_number = numbers > 0
        has_special = special_chars > 0
        
        # Check logic
        if length < self.MIN_LENGTH or length > self.MAX_LENGTH:
            return {
                "strength": "POOR",
                "score": 0,
                "feedback": f"Password must be between {self.MIN_LENGTH} and {self.MAX_LENGTH} characters."
            }

        if has_lower and has_upper and has_number and has_special:
            return {
                "strength": "VERY STRONG",
                "score": 100,
                "feedback": "Excellent password."
            }
        
        if has_lower and has_upper and has_number:
            return {
                "strength": "STRONG",
                "score": 75,
                "feedback": "Good password, try adding special characters for more security."
            }
            
        if has_lower and has_upper:
            return {
                "strength": "MEDIUM",
                "score": 50,
                "feedback": "Medium strength. Add numbers and special characters."
            }

        if has_lower or has_upper or has_number:
             return {
                "strength": "WEAK",
                "score": 25,
                "feedback": "Weak password. Use a mix of uppercase, lowercase, numbers, and special chars."
            }

        # Fallback for any other case (e.g. only special chars implementation quirk)
        return {
            "strength": "WEAK",
            "score": 10,
            "feedback": "Weak password structure."
        }


if __name__ == "__main__":
    import getpass
    print("Password Strength Analyzer")
    # Using getpass to avoid echoing input in terminal
    try:
        user_password = getpass.getpass("Enter your password: ")
    except Exception:
        # Fallback for environments where getpass might fail or non-interactive
        user_password = input("Enter your password: ")
        
    analyzer = PasswordStrengthAnalyzer()
    result = analyzer.check(user_password)
    
    print(f"\nResults:")
    print(f"Strength: {result['strength']}")
    print(f"Score: {result['score']}")
    print(f"Feedback: {result['feedback']}")
