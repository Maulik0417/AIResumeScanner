// index.js
const express = require("express");
const cors = require("cors");
const bodyParser = require("body-parser");
const axios = require("axios");
require("dotenv").config();

const app = express();
const PORT = process.env.PORT || 5001;

app.use(cors({
    origin: 'http://localhost:3000' // Replace with your frontend URL
}));

app.use(bodyParser.json());

app.post("/api/analyze", async (req, res) => {
    const { resumeText } = req.body;

    try {
        const response = await axios.post(
            // "https://api-inference.huggingface.co/models/gpt2",
            // "https://api-inference.huggingface.co/models/dbmdz/bert-large-cased-finetuned-conll03-english",
            "https://api-inference.huggingface.co/models/nlptown/bert-base-multilingual-uncased-sentiment",
            { inputs: resumeText },
            {
                headers: {
                    Authorization: `Bearer ${process.env.HUGGINGFACE_API_KEY}`,
                },
            }
        );

        res.json(response.data);
    } catch (error) {
        console.error("Error analyzing resume:", error);
        res.status(500).send("Error analyzing resume");
    }
});

app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
