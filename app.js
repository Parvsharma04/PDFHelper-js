const express = require('express');
const axios = require('axios');
const pdf = require('pdf-parse');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const Groq = require('groq-sdk');
const { MongoClient } = require('mongodb');
const { v4: uuidv4 } = require('uuid');
require('dotenv').config();

// --- Configuration ---
const GEMINI_API_KEY = process.env.GEMINI_KEY;
const GROQ_API_KEY = process.env.GROQ_API_KEY;
const MONGO_URI = process.env.MONGO_URI;
const MONGO_DB_NAME = "rag_db";
const MONGO_COLLECTION_NAME = "pdf_chunks";
const MONGO_VECTOR_INDEX_NAME = "default";

const EMBEDDING_DIMENSION = 768; // Note: Using text-embedding-004 which has 768 dimensions
const GEMINI_LLM_MODEL = "gemini-1.5-flash";
const GEMINI_EMBEDDING_MODEL = "text-embedding-004";
const GROQ_LLM_MODEL = "llama3-8b-8192";

const CHUNK_SIZE = 10000;
const CHUNK_OVERLAP = 2000;

// --- Initialize Express App ---
const app = express();
app.use(express.json());

// --- Logging Setup ---
const log = {
    info: (msg) => console.log(`[INFO] ${new Date().toISOString()} - ${msg}`),
    error: (msg) => console.error(`[ERROR] ${new Date().toISOString()} - ${msg}`),
    warning: (msg) => console.warn(`[WARN] ${new Date().toISOString()} - ${msg}`)
};

// --- Initialize External Clients ---
let genaiClient = null;
let mongoClient = null;
let mongoCollection = null;
let groqClient = null;

// Initialize Gemini AI
if (GEMINI_API_KEY) {
    try {
        genaiClient = new GoogleGenerativeAI(GEMINI_API_KEY);
        log.info("Gemini AI client configured successfully.");
    } catch (error) {
        log.error(`Failed to configure Gemini AI client: ${error.message}`);
    }
} else {
    log.error("GEMINI_KEY environment variable not set. Gemini services will not be available.");
}

// Initialize MongoDB
if (MONGO_URI) {
    (async () => {
        try {
            mongoClient = new MongoClient(MONGO_URI);
            await mongoClient.connect();
            await mongoClient.db('admin').command({ ping: 1 });
            mongoCollection = mongoClient.db(MONGO_DB_NAME).collection(MONGO_COLLECTION_NAME);
            log.info("MongoDB client and collection initialized.");
            log.info(`Connected to MongoDB database: '${MONGO_DB_NAME}', collection: '${MONGO_COLLECTION_NAME}'`);
        } catch (error) {
            log.error(`Failed to connect to MongoDB: ${error.message}`);
            mongoClient = null;
            mongoCollection = null;
        }
    })();
} else {
    log.error("MONGO_URI environment variable not set. MongoDB services will not be available.");
}

// Initialize Groq
if (GROQ_API_KEY) {
    try {
        groqClient = new Groq({ apiKey: GROQ_API_KEY });
        log.info("Groq client initialized.");
    } catch (error) {
        log.warning(`Failed to initialize Groq client: ${error.message}`);
    }
}

// --- Helper Functions ---

async function downloadPdf(url) {
    try {
        const response = await axios.get(url, { 
            responseType: 'arraybuffer',
            timeout: 30000
        });
        log.info(`Successfully downloaded PDF from ${url}`);
        return Buffer.from(response.data);
    } catch (error) {
        log.error(`Error downloading PDF from ${url}: ${error.message}`);
        throw new Error(`Failed to download PDF: ${error.message}. Please check the URL.`);
    }
}

async function extractTextFromPdf(pdfBuffer) {
    try {
        const data = await pdf(pdfBuffer);
        
        if (!data.text || data.text.trim().length === 0) {
            throw new Error("Could not extract any text from the PDF. It might be empty or in a format not supported for text extraction.");
        }

        log.info(`Extracted total characters from PDF: ${data.text.length}`);
        return data.text;
    } catch (error) {
        log.error(`Error extracting text from PDF: ${error.message}`);
        throw new Error(`Invalid or corrupted PDF content: ${error.message}`);
    }
}

function chunkText(text, chunkSize, chunkOverlap) {
    const chunks = [];
    let start = 0;
    
    while (start < text.length) {
        const end = start + chunkSize;
        const chunk = text.slice(start, Math.min(end, text.length));
        chunks.push(chunk);
        
        if (end >= text.length) break;
        start += chunkSize - chunkOverlap;
    }
    
    log.info(`Text chunked into ${chunks.length} pieces.`);
    return chunks;
}

async function generateEmbeddings(texts) {
    if (!genaiClient) {
        throw new Error("Gemini AI client is not initialized. Cannot generate embeddings.");
    }

    const filteredTexts = texts.filter(text => text && text.trim());
    if (filteredTexts.length === 0) {
        log.warning("No valid text chunks to generate embeddings for after filtering empty/whitespace.");
        return [];
    }

    log.info(`Attempting to generate embeddings for ${filteredTexts.length} total chunks using Gemini AI.`);

    try {
        const model = genaiClient.getGenerativeModel({ model: GEMINI_EMBEDDING_MODEL });
        const embeddings = [];
        
        // Process embeddings in batches to avoid rate limits
        const batchSize = 10;
        for (let i = 0; i < filteredTexts.length; i += batchSize) {
            const batch = filteredTexts.slice(i, i + batchSize);
            const batchPromises = batch.map(async (text) => {
                const result = await model.embedContent(text);
                return result.embedding.values;
            });
            
            const batchEmbeddings = await Promise.all(batchPromises);
            embeddings.push(...batchEmbeddings);
            
            // Add small delay between batches
            if (i + batchSize < filteredTexts.length) {
                await new Promise(resolve => setTimeout(resolve, 100));
            }
        }

        log.info(`Successfully generated ${embeddings.length} embeddings in total using Gemini AI.`);
        
        if (embeddings.length > 0 && embeddings[0].length !== EMBEDDING_DIMENSION) {
            log.error(`Generated embedding dimension (${embeddings[0].length}) does not match expected dimension (${EMBEDDING_DIMENSION}).`);
            throw new Error(`Embedding dimension mismatch: Expected ${EMBEDDING_DIMENSION}, got ${embeddings[0].length}.`);
        }

        return embeddings;
    } catch (error) {
        log.error(`Gemini AI embedding generation failed: ${error.message}`);
        throw new Error(`Failed to generate embeddings: ${error.message}`);
    }
}

async function storeEmbeddingsInMongodb(collection, textChunks, embeddings, documentId) {
    if (!collection) {
        throw new Error("MongoDB collection is not initialized or usable.");
    }

    if (textChunks.length !== embeddings.length) {
        log.error(`Mismatched lengths of text_chunks (${textChunks.length}) and embeddings (${embeddings.length}) for storage.`);
        throw new Error("Internal error: Mismatch between chunks and embeddings. Data integrity issue.");
    }

    const documentsToInsert = textChunks.map((chunk, index) => ({
        text_chunk: chunk,
        embedding: embeddings[index],
        document_id: documentId,
        chunk_index: index,
        timestamp: Date.now()
    }));

    if (documentsToInsert.length === 0) {
        log.warning("No documents to insert into MongoDB. This might mean no text was extracted or no embeddings were generated.");
        return;
    }

    try {
        const result = await collection.insertMany(documentsToInsert);
        log.info(`Successfully inserted ${result.insertedCount} documents into MongoDB for document ${documentId}.`);
    } catch (error) {
        log.error(`Error inserting embeddings into MongoDB: ${error.message}`);
        throw new Error(`Failed to store embeddings: ${error.message}`);
    }
}

async function queryMongodbAndRag(collection, question, llmClientType = "gemini", topK = 3) {
    if (!collection) {
        throw new Error("MongoDB collection is not initialized or usable.");
    }
    if (!genaiClient) {
        throw new Error("Gemini AI client is not initialized. Cannot generate query embedding.");
    }
    if (llmClientType === "gemini" && !GEMINI_API_KEY) {
        throw new Error("GEMINI_KEY is not set for Gemini LLM.");
    }
    if (llmClientType === "groq" && !groqClient) {
        throw new Error("Groq client not initialized for Groq LLM. Cannot proceed with Groq.");
    }

    try {
        // 1. Generate embedding for the user's question
        let queryEmbedding;
        try {
            const model = genaiClient.getGenerativeModel({ model: GEMINI_EMBEDDING_MODEL });
            const result = await model.embedContent(question);
            queryEmbedding = result.embedding.values;
        } catch (error) {
            log.error(`Error generating query embedding: ${error.message}`);
            throw new Error(`Query embedding generation failed: ${error.message}`);
        }

        // 2. Query MongoDB using Atlas Vector Search
        const atlasVectorSearchPipeline = [
            {
                $vectorSearch: {
                    queryVector: queryEmbedding,
                    path: "embedding",
                    numCandidates: topK * 40,
                    limit: topK,
                    index: MONGO_VECTOR_INDEX_NAME
                }
            },
            {
                $project: {
                    text_chunk: 1,
                    _id: 0,
                    score: { $meta: "vectorSearchScore" }
                }
            }
        ];

        const queryResults = await collection.aggregate(atlasVectorSearchPipeline).toArray();
        log.info(`Vector search results: ${JSON.stringify(queryResults)}`);

        const contextChunks = queryResults
            .filter(result => result.text_chunk)
            .map(result => result.text_chunk);

        let contextString;
        if (contextChunks.length === 0) {
            log.warning(`No relevant context found in MongoDB for question: '${question}'. Answering without specific context.`);
            contextString = "No specific context found in the document. Please note: The answer might be general without document-specific context.";
        } else {
            contextString = contextChunks.join("\n\n");
            log.info(`Retrieved ${contextChunks.length} context chunks for question: '${question}'`);
        }

        // 3. Construct the prompt for the LLM
        const prompt = `You are an AI assistant. Use the following context to answer the question.
If the answer is not explicitly available in the provided context, state that you don't know or that the information is not in the document.

Context:
${contextString}

Question: ${question}

Answer:`;

        // 4. Call the chosen LLM
        let answer;
        if (llmClientType === "gemini") {
            if (!GEMINI_API_KEY) {
                throw new Error("GEMINI_KEY is not set for Gemini LLM.");
            }
            try {
                const model = genaiClient.getGenerativeModel({ model: GEMINI_LLM_MODEL });
                const result = await model.generateContent(prompt);
                answer = result.response.text();
                log.info(`Gemini LLM answered question: '${question}'`);
            } catch (error) {
                log.error(`Error calling Gemini API: ${error.message}`);
                throw new Error(`Gemini LLM call failed: ${error.message}`);
            }
        } else if (llmClientType === "groq") {
            if (!groqClient) {
                throw new Error("Groq client not initialized. Cannot use Groq LLM.");
            }
            try {
                const chatCompletion = await groqClient.chat.completions.create({
                    messages: [
                        {
                            role: "user",
                            content: prompt,
                        }
                    ],
                    model: GROQ_LLM_MODEL,
                    temperature: 0.7,
                    max_tokens: 500,
                });
                answer = chatCompletion.choices[0].message.content;
                log.info(`Groq LLM answered question: '${question}'`);
            } catch (error) {
                log.error(`Error calling Groq API: ${error.message}`);
                throw new Error(`Groq API call failed: ${error.message}`);
            }
        } else {
            throw new Error("Invalid LLM client type specified. Must be 'gemini' or 'groq'.");
        }

        return answer;

    } catch (error) {
        log.error(`Error during RAG process for question '${question}': ${error.message}`);
        throw new Error(`Failed to answer question: ${error.message}`);
    }
}

// --- Express Routes ---

app.get('/', (req, res) => {
    res.json({ works: "yes" });
});

app.post('/hackrx/run', async (req, res) => {
    try {
        const { documents, questions } = req.body;

        // Validation
        if (!documents || !questions || !Array.isArray(questions)) {
            return res.status(400).json({ 
                error: "Invalid request format. Expected 'documents' (string) and 'questions' (array)." 
            });
        }

        if (!GEMINI_API_KEY && !groqClient) {
            return res.status(500).json({ 
                error: "Neither Gemini nor Groq LLM client is initialized. Cannot answer questions." 
            });
        }

        if (!mongoCollection) {
            return res.status(500).json({ 
                error: "MongoDB client or collection failed to initialize. Check your MONGO_URI and Atlas setup." 
            });
        }

        if (!GEMINI_API_KEY) {
            return res.status(500).json({ 
                error: "Gemini API key is not set. Cannot generate embeddings." 
            });
        }

        log.info(`Received request to process PDF from: ${documents}`);
        log.info(`Questions received: ${JSON.stringify(questions)}`);

        // Process PDF
        const pdfContent = await downloadPdf(documents);
        const extractedText = await extractTextFromPdf(pdfContent);
        const textChunks = chunkText(extractedText, CHUNK_SIZE, CHUNK_OVERLAP);

        const filteredTextChunks = textChunks.filter(chunk => chunk && chunk.trim());

        if (filteredTextChunks.length === 0) {
            return res.status(400).json({ 
                error: "No usable text chunks extracted from the PDF after filtering." 
            });
        }

        // Generate embeddings
        const embeddings = await generateEmbeddings(filteredTextChunks);

        if (embeddings.length > 0 && embeddings[0].length !== EMBEDDING_DIMENSION) {
            return res.status(500).json({ 
                error: `Generated embedding dimension (${embeddings[0].length}) does not match expected dimension (${EMBEDDING_DIMENSION}).` 
            });
        }

        if (filteredTextChunks.length !== embeddings.length) {
            log.error(`Critical Mismatch: chunks (${filteredTextChunks.length}) vs embeddings (${embeddings.length})`);
            return res.status(500).json({ 
                error: "Internal data processing error: Mismatch in chunk-embedding count." 
            });
        }

        // Store embeddings
        const documentId = uuidv4();
        await storeEmbeddingsInMongodb(mongoCollection, filteredTextChunks, embeddings, documentId);

        // Answer questions
        const answers = [];
        for (const question of questions) {
            const answer = await queryMongodbAndRag(mongoCollection, question, "gemini");
            answers.push(answer);
        }

        log.info("Successfully processed PDF and answered all questions.");
        log.info(`Answers: ${JSON.stringify(answers)}`);
        
        res.json({ answers });

    } catch (error) {
        log.error(`API Error: ${error.message}`);
        res.status(500).json({ error: error.message });
    }
});

// --- Error Handling Middleware ---
app.use((error, req, res, next) => {
    log.error(`Unhandled error: ${error.message}`);
    res.status(500).json({ error: "Internal server error" });
});

// --- Start Server ---
const PORT = 3000;
app.listen(PORT, () => {
    log.info(`Server running on port ${PORT}`);
});

module.exports = app;
