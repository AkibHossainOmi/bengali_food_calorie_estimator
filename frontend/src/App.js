import { BrowserRouter as Router, Routes, Route, Navigate } from "react-router-dom";
import React, { useContext } from "react";

import Navbar from "./components/Navbar";
import Predictor from "./components/Predictor";
import Home from "./components/Home";

import RegisterForm from "./components/RegisterForm";
import LoginForm from "./components/LoginForm";
import EmailVerification from "./components/EmailVerification";
import PasswordResetRequest from "./components/PasswordResetRequest";
import PasswordResetConfirm from "./components/PasswordResetConfirm";

import { AuthProvider, AuthContext } from "./context/AuthContext";

function AppRoutes() {
  const { loggedIn, handleLogin, handleLogout } = useContext(AuthContext);

  return (
    <>
      <Navbar loggedIn={loggedIn} onLogout={handleLogout} />

      <Routes>
        <Route path="/" element={<Home />} />

        {/* Protected route: only logged-in users can access /predict */}
        <Route
          path="/predict"
          element={loggedIn ? <Predictor /> : <Navigate to="/login" replace />}
        />

        {/* Public routes */}
        <Route
          path="/register"
          element={loggedIn ? <Navigate to="/" replace /> : <RegisterForm />}
        />
        <Route
          path="/login"
          element={loggedIn ? <Navigate to="/" replace /> : <LoginForm onLogin={handleLogin} />}
        />

        {/* Email verification page */}
        <Route path="/verify-email" element={<EmailVerification />} />

        {/* Password reset pages */}
        <Route path="/password-reset-request" element={<PasswordResetRequest />} />
        <Route path="/password-reset-confirm" element={<PasswordResetConfirm />} />

        {/* Catch-all route */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </>
  );
}

function App() {
  return (
    <AuthProvider>
      <Router>
        <AppRoutes />
      </Router>
    </AuthProvider>
  );
}

export default App;
