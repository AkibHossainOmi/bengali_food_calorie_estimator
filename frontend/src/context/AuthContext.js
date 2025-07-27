import React, { createContext, useState, useEffect } from "react";

export const AuthContext = createContext({
  loggedIn: false,
  handleLogin: () => {},
  handleLogout: () => {},
});

export const AuthProvider = ({ children }) => {
  const [loggedIn, setLoggedIn] = useState(false);

  useEffect(() => {
    const token = localStorage.getItem("access_token");
    setLoggedIn(!!token);
  }, []);

  const handleLogin = () => setLoggedIn(true);

  const handleLogout = () => {
    localStorage.removeItem("access_token");
    setLoggedIn(false);
  };

  return (
    <AuthContext.Provider value={{ loggedIn, handleLogin, handleLogout }}>
      {children}
    </AuthContext.Provider>
  );
};
