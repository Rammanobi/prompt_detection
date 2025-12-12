// Import the functions you need from the SDKs you need
import { initializeApp } from "https://www.gstatic.com/firebasejs/10.7.1/firebase-app.js";
import { getAuth } from "https://www.gstatic.com/firebasejs/10.7.1/firebase-auth.js";

// TODO: Replace with your actual Firebase project configuration
// You can find this in your Firebase Console -> Project Settings -> General -> Your Apps -> SDK Setup/Configuration
const firebaseConfig = {
  apiKey: "AIzaSyAYbnPr3-XVj02Qonz9vXgCE6UUYKyQwsc",
  authDomain: "my-project-id-45.firebaseapp.com",
  projectId: "my-project-id-45",
  storageBucket: "my-project-id-45.firebasestorage.app",
  messagingSenderId: "267340528308",
  appId: "1:267340528308:web:51eaa763f87f0c418675e1",
  measurementId: "G-X6H4SSH07G"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const auth = getAuth(app);

export { auth };
