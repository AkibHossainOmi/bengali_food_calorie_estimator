import React, { createContext, useState, useEffect } from "react";

export const AuthContext = createContext({
  loggedIn: false,
  handleLogin: () => {},
  handleLogout: () => {},
});

export const AuthProvider = ({ children }) => {
  const [loggedIn, setLoggedIn] = useState(false);
  const [initialized, setInitialized] = useState(false); // Track initialization

  useEffect(() => {
    // Verify token validity when app loads
    const token = localStorage.getItem("access_token");
    if (token) {
      // Add token validation logic here if needed
      setLoggedIn(true);
    }
    setInitialized(true);
  }, []);

  const handleLogin = (token) => {
    localStorage.setItem("access_token", token);
    setLoggedIn(true);
  };

  const handleLogout = () => {
    localStorage.removeItem("access_token");
    setLoggedIn(false);
  };

  // Don't render children until auth state is initialized
  if (!initialized) {
    return <div>Loading...</div>; // Or a loading spinner
  }

  return (
    <AuthContext.Provider value={{ loggedIn, handleLogin, handleLogout }}>
      {children}
    </AuthContext.Provider>
  );
};
