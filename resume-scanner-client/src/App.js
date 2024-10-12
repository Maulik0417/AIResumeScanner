// src/App.js
import React, { useState } from "react";
import axios from "axios";

const App = () => {
    const [resumeText, setResumeText] = useState("");
    const [analysisResult, setAnalysisResult] = useState(null);

    const handleSubmit = async (e) => {
        e.preventDefault();
        try {
            const response = await axios.post("http://localhost:5001/api/analyze", {
                resumeText,
            });
            setAnalysisResult(response.data);
        } catch (error) {
            console.error("Error submitting resume:", error);
        }
    };

    return (
        <div style={{ padding: "20px" }}>
            <h1>AI Resume Scanner</h1>
            <form onSubmit={handleSubmit}>
                <textarea
                    rows="10"
                    cols="50"
                    value={resumeText}
                    onChange={(e) => setResumeText(e.target.value)}
                    placeholder="Paste your resume text here..."
                />
                <br />
                <button type="submit">Analyze Resume</button>
            </form>

            {analysisResult && (
                <div>
                    <h2>Analysis Result</h2>
                    <pre>{JSON.stringify(analysisResult, null, 2)}</pre>
                </div>
            )}
        </div>
    );
};

export default App;
