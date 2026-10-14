import pickle
from flask import Flask, render_template_string, request
import numpy as np

app = Flask(__name__)

# Load the trained Naive Bayes model
with open('naive.pkl', 'rb') as f:
    model = pickle.load(f)

# HTML Template with Tailwind CSS for a modern, attractive layout
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Employee Prediction Portal</title>
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gradient-to-br from-slate-900 via-indigo-950 to-slate-900 min-h-screen text-slate-100 flex items-center justify-center p-4">
    <div class="max-w-2xl w-full bg-slate-900/80 backdrop-blur-md border border-slate-800 rounded-2xl shadow-2xl overflow-hidden p-8">
        
        <!-- Header -->
        <div class="text-center mb-8">
            <h1 class="text-3xl font-extrabold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 to-cyan-400">
                Employee Predictor
            </h1>
            <p class="text-slate-400 text-sm mt-2">Powered by your Naive Bayes Machine Learning Model</p>
        </div>

        <!-- Prediction Result Card -->
        {% if prediction is not none %}
        <div class="mb-8 p-4 rounded-xl border {% if prediction == 1 %} bg-emerald-950/50 border-emerald-500/50 text-emerald-300 {% else %} bg-rose-950/50 border-rose-500/50 text-rose-300 {% endif %} flex items-center justify-between shadow-lg">
            <div>
                <span class="text-xs uppercase tracking-wider font-semibold opacity-75">Prediction Result</span>
                <p class="text-xl font-bold mt-1">
                    {% if prediction == 1 %}
                        🎉 Employee is likely to leave / match criteria (Class: 1)
                    {% else %}
                        🛡️ Employee is likely to stay / not match criteria (Class: 0)
                    {% endif %}
                </p>
            </div>
        </div>
        {% endif %}

        <!-- Input Form -->
        <form method="POST" class="grid grid-cols-1 md:grid-cols-2 gap-6">
            
            <!-- Education -->
            <div>
                <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Education</label>
                <select name="Education" required class="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5 text-slate-200 focus:outline-none focus:border-indigo-500 transition">
                    <option value="0">Bachelors</option>
                    <option value="1">Masters</option>
                    <option value="2">PHd</option>
                </select>
                <span class="text-[10px] text-slate-500 mt-1 block">Ensure this matches your label encoding</span>
            </div>

            <!-- Joining Year -->
            <div>
                <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Joining Year</label>
                <input type="number" name="JoiningYear" placeholder="e.g. 2018" required class="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5 text-slate-200 focus:outline-none focus:border-indigo-500 transition">
            </div>

            <!-- City -->
            <div>
                <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">City</label>
                <select name="City" required class="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5 text-slate-200 focus:outline-none focus:border-indigo-500 transition">
                    <option value="0">Bangalore</option>
                    <option value="1">Pune</option>
                    <option value="2">New Delhi</option>
                </select>
            </div>

            <!-- Payment Tier -->
            <div>
                <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Payment Tier</label>
                <select name="PaymentTier" required class="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5 text-slate-200 focus:outline-none focus:border-indigo-500 transition">
                    <option value="1">Tier 1</option>
                    <option value="2">Tier 2</option>
                    <option value="3">Tier 3</option>
                </select>
            </div>

            <!-- Age -->
            <div>
                <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Age</label>
                <input type="number" name="Age" placeholder="e.g. 28" required class="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5 text-slate-200 focus:outline-none focus:border-indigo-500 transition">
            </div>

            <!-- Gender -->
            <div>
                <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Gender</label>
                <select name="Gender" required class="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5 text-slate-200 focus:outline-none focus:border-indigo-500 transition">
                    <option value="0">Male</option>
                    <option value="1">Female</option>
                </select>
            </div>

            <!-- Ever Benched -->
            <div>
                <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Ever Benched</label>
                <select name="EverBenched" required class="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5 text-slate-200 focus:outline-none focus:border-indigo-500 transition">
                    <option value="0">No</option>
                    <option value="1">Yes</option>
                </select>
            </div>

            <!-- Experience in Current Domain -->
            <div>
                <label class="block text-xs font-semibold uppercase tracking-wider text-slate-400 mb-2">Experience in Current Domain</label>
                <input type="number" name="ExperienceInCurrentDomain" placeholder="e.g. 3" required class="w-full bg-slate-800 border border-slate-700 rounded-lg px-4 py-2.5 text-slate-200 focus:outline-none focus:border-indigo-500 transition">
            </div>

            <!-- Submit Button -->
            <div class="md:col-span-2 mt-4">
                <button type="submit" class="w-full py-3 px-6 bg-gradient-to-r from-indigo-500 to-cyan-500 hover:from-indigo-600 hover:to-cyan-600 text-white font-semibold rounded-lg shadow-lg transition duration-200 transform active:scale-[0.99]">
                    Run Prediction
                </button>
            </div>
        </form>
    </div>
</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        try:
            # Extract features in the exact order expected by the model[cite: 1]
            features = [
                float(request.form['Education']),
                float(request.form['JoiningYear']),
                float(request.form['City']),
                float(request.form['PaymentTier']),
                float(request.form['Age']),
                float(request.form['Gender']),
                float(request.form['EverBenched']),
                float(request.form['ExperienceInCurrentDomain'])
            ]
            
            # Reshape into 2D array for sklearn prediction
            final_features = np.array([features])
            prediction = int(model.predict(final_features)[0])
        except Exception as e:
            print(f"Error during prediction: {e}")
            
    return render_template_string(HTML_TEMPLATE, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
