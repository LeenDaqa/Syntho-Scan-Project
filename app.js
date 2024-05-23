const express = require('express');
const multer = require('multer');
const bodyParser = require('body-parser');
const cors = require('cors');
const path = require('path');
const session = require('express-session');
const fs = require('fs');
const axios = require('axios');

const app = express();
const port = 3000;

app.use(cors());
app.use(bodyParser.json());
app.use(bodyParser.urlencoded({ extended: true }));
app.use(express.static(path.join(__dirname, 'public')));

app.use(session({
    secret: 'dummy-secret-key',
    resave: false,
    saveUninitialized: true,
    cookie: { secure: false } // Set to true if using HTTPS
}));

const upload = multer({ dest: 'uploads/' });

const dummyUser = {
    username: 'testuser',
    password: 'password123'
};

// Serve login page
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'login.html'));
});

// Handle login
app.post('/login', (req, res) => {
    const { username, password } = req.body;
    if (username === dummyUser.username && password === dummyUser.password) {
        req.session.user = dummyUser;
        res.status(200).send('Login successful');
    } else {
        res.status(401).send('Invalid username or password');
    }
});

// Serve transform page
app.get('/upload.html', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'upload.html'));
});

// Serve results page
app.get('/mriCt.html', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'mriCt.html'));
});

let resultData = {}; // To store the results temporarily

// Handle image transformation (A2B)
app.post('/transform', upload.single('image'), async (req, res) => {
    const imagePath = req.file.path;

    try {
        const fileBuffer = fs.readFileSync(imagePath);
        const response = await axios.post('http://localhost:8001/transform', fileBuffer, {
            headers: {
                'Content-Type': 'application/octet-stream',
            }
        });

        resultData = response.data;
        resultData.uploadedImage = fs.readFileSync(imagePath, { encoding: 'base64' }); // Store the uploaded image
        res.json(resultData);
        fs.unlinkSync(imagePath); // Clean up the uploaded file
    } catch (error) {
        console.error(`Error transforming image: ${error}`);
        res.status(500).send('Error transforming image');
    }
});

// Handle image transformation (B2A)
app.post('/transform2', upload.single('image'), async (req, res) => {
    const imagePath = req.file.path;

    try {
        const fileBuffer = fs.readFileSync(imagePath);
        const response = await axios.post('http://localhost:8001/transform2', fileBuffer, {
            headers: {
                'Content-Type': 'application/octet-stream',
            }
        });

        resultData = response.data;
        resultData.uploadedImage = fs.readFileSync(imagePath, { encoding: 'base64' }); // Store the uploaded image
        res.json(resultData);
        fs.unlinkSync(imagePath); // Clean up the uploaded file
    } catch (error) {
        console.error(`Error transforming image: ${error}`);
        res.status(500).send('Error transforming image');
    }
});

// Provide the result data
app.get('/get-result', (req, res) => {
    res.json(resultData);
});

app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});
