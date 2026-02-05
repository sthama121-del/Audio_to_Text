"""
AI-Powered Resume Tailoring Tool
Interactive GUI version - Paste JD and specifications, get tailored .docx resume
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import anthropic
import os
from pathlib import Path
import subprocess
import json
from datetime import datetime

class ResumeTailorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Resume Tailor - Srikanth Thamagonda")
        self.root.geometry("1200x800")
        
        # Configure style
        style = ttk.Style()
        style.configure('Title.TLabel', font=('Calibri', 16, 'bold'))
        style.configure('Section.TLabel', font=('Calibri', 11, 'bold'))
        
        self.setup_ui()
        self.load_saved_data()
        
    def setup_ui(self):
        """Create the UI layout"""
        
        # Main container with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        main_frame.rowconfigure(4, weight=1)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="🎯 AI-Powered Resume Tailoring Tool",
            style='Title.TLabel'
        )
        title_label.grid(row=0, column=0, pady=(0, 20))
        
        # API Key Section
        api_frame = ttk.LabelFrame(main_frame, text="API Configuration", padding="10")
        api_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        api_frame.columnconfigure(1, weight=1)
        
        ttk.Label(api_frame, text="Anthropic API Key:").grid(row=0, column=0, sticky=tk.W, padx=(0, 10))
        self.api_key_entry = ttk.Entry(api_frame, width=60, show="*")
        self.api_key_entry.grid(row=0, column=1, sticky=(tk.W, tk.E))
        
        show_key_btn = ttk.Button(api_frame, text="Show", command=self.toggle_api_key, width=8)
        show_key_btn.grid(row=0, column=2, padx=(5, 0))
        
        # Job Description Section
        jd_frame = ttk.LabelFrame(main_frame, text="📄 Job Description", padding="10")
        jd_frame.grid(row=2, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        jd_frame.columnconfigure(0, weight=1)
        jd_frame.rowconfigure(0, weight=1)
        
        self.jd_text = scrolledtext.ScrolledText(
            jd_frame, 
            wrap=tk.WORD, 
            width=100, 
            height=15,
            font=('Calibri', 10)
        )
        self.jd_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Master Specifications Section
        spec_frame = ttk.LabelFrame(main_frame, text="⚙️ Master Specifications", padding="10")
        spec_frame.grid(row=4, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        spec_frame.columnconfigure(0, weight=1)
        spec_frame.rowconfigure(0, weight=1)
        
        self.spec_text = scrolledtext.ScrolledText(
            spec_frame, 
            wrap=tk.WORD, 
            width=100, 
            height=15,
            font=('Calibri', 10)
        )
        self.spec_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Action Buttons
        button_frame = ttk.Frame(main_frame, padding="10")
        button_frame.grid(row=5, column=0, sticky=(tk.W, tk.E))
        
        self.generate_btn = ttk.Button(
            button_frame, 
            text="🚀 Generate Tailored Resume",
            command=self.generate_resume,
            style='Accent.TButton'
        )
        self.generate_btn.pack(side=tk.LEFT, padx=5)
        
        save_config_btn = ttk.Button(
            button_frame, 
            text="💾 Save Configuration",
            command=self.save_config
        )
        save_config_btn.pack(side=tk.LEFT, padx=5)
        
        load_config_btn = ttk.Button(
            button_frame, 
            text="📂 Load Configuration",
            command=self.load_config
        )
        load_config_btn.pack(side=tk.LEFT, padx=5)
        
        clear_btn = ttk.Button(
            button_frame, 
            text="🗑️ Clear All",
            command=self.clear_all
        )
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready. Paste your Job Description and Master Specifications, then click Generate.")
        status_bar = ttk.Label(
            main_frame, 
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        status_bar.grid(row=6, column=0, sticky=(tk.W, tk.E), pady=(10, 0))
        
    def toggle_api_key(self):
        """Toggle API key visibility"""
        current = self.api_key_entry.cget("show")
        self.api_key_entry.config(show="" if current == "*" else "*")
        
    def save_config(self):
        """Save current configuration to JSON file"""
        config = {
            "api_key": self.api_key_entry.get(),
            "master_specifications": self.spec_text.get("1.0", tk.END).strip()
        }
        
        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile="resume_config.json"
        )
        
        if filepath:
            with open(filepath, 'w') as f:
                json.dump(config, f, indent=2)
            messagebox.showinfo("Success", f"Configuration saved to:\n{filepath}")
            
    def load_config(self):
        """Load configuration from JSON file"""
        filepath = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                with open(filepath, 'r') as f:
                    config = json.load(f)
                
                self.api_key_entry.delete(0, tk.END)
                self.api_key_entry.insert(0, config.get("api_key", ""))
                
                self.spec_text.delete("1.0", tk.END)
                self.spec_text.insert("1.0", config.get("master_specifications", ""))
                
                messagebox.showinfo("Success", "Configuration loaded successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load configuration:\n{str(e)}")
                
    def load_saved_data(self):
        """Load previously saved data if exists"""
        config_file = Path.home() / ".resume_tailor_config.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                self.api_key_entry.insert(0, config.get("api_key", ""))
                self.spec_text.insert("1.0", config.get("master_specifications", ""))
            except:
                pass
                
    def save_data(self):
        """Auto-save configuration"""
        config_file = Path.home() / ".resume_tailor_config.json"
        config = {
            "api_key": self.api_key_entry.get(),
            "master_specifications": self.spec_text.get("1.0", tk.END).strip()
        }
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
            
    def clear_all(self):
        """Clear all input fields"""
        if messagebox.askyesno("Confirm", "Clear all fields?"):
            self.jd_text.delete("1.0", tk.END)
            
    def generate_resume(self):
        """Generate tailored resume using Claude API"""
        
        # Validate inputs
        api_key = self.api_key_entry.get().strip()
        jd_content = self.jd_text.get("1.0", tk.END).strip()
        spec_content = self.spec_text.get("1.0", tk.END).strip()
        
        if not api_key:
            messagebox.showerror("Error", "Please enter your Anthropic API key")
            return
            
        if not jd_content:
            messagebox.showerror("Error", "Please paste the Job Description")
            return
            
        if not spec_content:
            messagebox.showerror("Error", "Please paste the Master Specifications")
            return
        
        # Auto-save configuration
        self.save_data()
        
        # Disable button during generation
        self.generate_btn.config(state='disabled')
        self.status_var.set("🔄 Generating tailored resume... Please wait...")
        self.root.update()
        
        try:
            # Create the prompt for Claude
            prompt = f"""You are an expert resume tailoring assistant. I need you to tailor a resume based on the following:

<job_description>
{jd_content}
</job_description>

<master_specifications>
{spec_content}
</master_specifications>

Please create a tailored resume following ALL the specifications exactly. The resume should:
1. Match the job description requirements while following formatting rules
2. Use Calibri font throughout
3. Apply proper color coding (Dark Green 1 for headers)
4. Maintain exact margins (0.7" all sides)
5. Keep experience totals as specified (15+ years IT, 3 years AI)
6. Follow all bullet point and alignment rules
7. Map cloud technologies appropriately (Azure only)
8. Not exceed 3 pages

Output the resume content in a structured format that can be converted to a .docx file.

IMPORTANT: After the resume, provide a brief 3-5 line summary of the tailoring changes made."""

            # Call Claude API
            client = anthropic.Anthropic(api_key=api_key)
            
            self.status_var.set("🤖 Calling Claude API...")
            self.root.update()
            
            message = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=8000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            response_text = message.content[0].text
            
            # Save the response
            output_dir = Path.home() / "Desktop" / "Tailored_Resumes"
            output_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            txt_filename = output_dir / f"Srikanth_Thamagonda_Resume_{timestamp}.txt"
            
            with open(txt_filename, 'w', encoding='utf-8') as f:
                f.write(response_text)
            
            self.status_var.set("✅ Resume generated successfully!")
            
            result_msg = f"""Resume tailored successfully!

📁 Saved to: {txt_filename}

Note: The resume is currently in text format. For a professional .docx format with proper formatting, you'll need to:
1. Use the text as a template
2. Apply formatting in Word (Calibri, colors, margins)
3. Or use a Word generation script

Would you like to open the output folder?"""
            
            if messagebox.askyesno("Success", result_msg):
                if os.name == 'nt':  # Windows
                    os.startfile(output_dir)
                elif os.name == 'posix':  # Mac/Linux
                    subprocess.run(['open', output_dir])
                    
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate resume:\n{str(e)}")
            self.status_var.set("❌ Error occurred during generation")
            
        finally:
            self.generate_btn.config(state='normal')


def main():
    """Main entry point"""
    root = tk.Tk()
    app = ResumeTailorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
