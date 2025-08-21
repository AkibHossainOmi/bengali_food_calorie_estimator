import { BrowserRouter as Router, Routes, Route, Navigate, useLocation } from "react-router-dom";
import React, { useContext } from "react";

import Navbar from "./components/Navbar";
import Dashboard from "./components/Dashboard";
import Home from "./components/Home";
import RegisterForm from "./components/RegisterForm";
import LoginForm from "./components/LoginForm";
import EmailVerification from "./components/EmailVerification";
import PasswordResetRequest from "./components/PasswordResetRequest";
import PasswordResetConfirm from "./components/PasswordResetConfirm";
import About from "./components/About";
import Footer from "./components/Footer";

import { AuthProvider, AuthContext } from "./context/AuthContext";
import Profile from "./components/Profile";

// Create a separate ProtectedRoute component for better reusability
function ProtectedRoute({ children }) {
  const { loggedIn } = useContext(AuthContext);
  const location = useLocation();

  if (!loggedIn) {
    // Pass current location to redirect back after login
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  return children;
}

function AppRoutes() {
  const { loggedIn, handleLogin, handleLogout } = useContext(AuthContext);

  return (
    <>
      <Navbar loggedIn={loggedIn} onLogout={handleLogout} />

      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/about" element={<About />} />

        {/* Protected routes */}
        <Route
          path="/dashboard"
          element={
            <ProtectedRoute>
              <Dashboard />
            </ProtectedRoute>
          }
        />

        <Route
          path="/profile"
          element={
            <ProtectedRoute>
              <Profile />
            </ProtectedRoute>
          }
        />

        {/* Auth routes - only accessible when not logged in */}
        <Route
          path="/register"
          element={loggedIn ? <Navigate to="/" replace /> : <RegisterForm />}
        />
        <Route
          path="/login"
          element={
            loggedIn ? (
              <Navigate to="/" replace />
            ) : (
              <LoginForm onLogin={handleLogin} />
            )
          }
        />

        {/* Email verification page */}
        <Route path="/verify-email" element={<EmailVerification />} />

        {/* Password reset pages */}
        <Route path="/password-reset-request" element={<PasswordResetRequest />} />
        <Route path="/password-reset-confirm" element={<PasswordResetConfirm />} />

        {/* Catch-all route */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
      
      <Footer />
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
