import React, { createContext, useState, useEffect } from "react";
import { profileAPI } from "../utils/apiService"; // assuming you have an API to fetch user profile

export const AuthContext = createContext({
  loggedIn: false,
  user: null,
  handleLogin: () => {},
  handleLogout: () => {},
  fetchUserProfile: async () => {},
});

export const AuthProvider = ({ children }) => {
  const [loggedIn, setLoggedIn] = useState(false);
  const [user, setUser] = useState(null);
  const [initialized, setInitialized] = useState(false);

  useEffect(() => {
    const token = localStorage.getItem("access_token");
    if (token) {
      setLoggedIn(true);
      fetchUserProfile();
    } else {
      setInitialized(true);
    }
  }, []);

  const fetchUserProfile = async () => {
    try {
      const res = await profileAPI.getProfile(); // fetch user profile from backend
      setUser(res.data);
    } catch (err) {
      console.error("Failed to fetch user profile:", err);
      setUser(null);
      setLoggedIn(false);
    } finally {
      setInitialized(true);
    }
  };

  const handleLogin = async (token) => {
    localStorage.setItem("access_token", token);
    setLoggedIn(true);
    await fetchUserProfile(); // fetch user info after login
  };

  const handleLogout = () => {
    localStorage.removeItem("access_token");
    setLoggedIn(false);
    setUser(null);
  };

  if (!initialized) {
    return <div>Loading...</div>; // Or spinner
  }

  return (
    <AuthContext.Provider value={{ loggedIn, user, handleLogin, handleLogout, fetchUserProfile }}>
      {children}
    </AuthContext.Provider>
  );
};
