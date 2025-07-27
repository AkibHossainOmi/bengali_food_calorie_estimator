import React, { useContext } from "react";
import { Link, useLocation } from "react-router-dom";
import { AuthContext } from "../context/AuthContext";

export default function Navbar() {
  const location = useLocation();
  const { loggedIn, handleLogout } = useContext(AuthContext);

  const linkClasses = (path) =>
    `px-4 py-2 rounded-md font-medium transition-all duration-200 flex items-center ${
      location.pathname === path
        ? "bg-gray-100 text-gray-900 font-semibold"
        : "text-gray-600 hover:bg-gray-50 hover:text-gray-800"
    }`;

  return (
    <nav className="bg-white shadow-sm text-gray-700 px-6 py-3 sticky top-0 z-50 border-b border-gray-200">
      <div className="max-w-7xl mx-auto flex items-center justify-between">
        {/* Left side - Navigation Links */}
        <div className="flex items-center space-x-4">
          <Link to="/" className={linkClasses("/")}>
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
              <path d="M10.707 2.293a1 1 0 00-1.414 0l-7 7a1 1 0 001.414 1.414L4 10.414V17a1 1 0 001 1h2a1 1 0 001-1v-2a1 1 0 011-1h2a1 1 0 011 1v2a1 1 0 001 1h2a1 1 0 001-1v-6.586l.293.293a1 1 0 001.414-1.414l-7-7z" />
            </svg>
            Home
          </Link>
          {loggedIn && (
            <Link to="/predict" className={linkClasses("/predict")}>
              <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5 mr-2" viewBox="0 0 20 20" fill="currentColor">
                <path fillRule="evenodd" d="M12 7a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0V8.414l-4.293 4.293a1 1 0 01-1.414 0L8 10.414l-4.293 4.293a1 1 0 01-1.414-1.414l5-5a1 1 0 011.414 0L11 10.586 14.586 7H12z" clipRule="evenodd" />
              </svg>
              Predict
            </Link>
          )}
        </div>

        {/* Right side - Auth Links */}
        <div className="flex items-center space-x-3">
          {loggedIn ? (
            <button
              onClick={handleLogout}
              className="px-4 py-2 rounded-md font-medium text-white bg-gray-700 hover:bg-gray-800 transition-colors duration-200 flex items-center"
            >
              Logout
            </button>
          ) : (
            <>
              <Link to="/login" className={`${linkClasses("/login")} border border-gray-200 hover:border-gray-300`}>
                Login
              </Link>
              <Link 
                to="/register" 
                className="px-4 py-2 rounded-md font-medium text-white bg-gray-900 hover:bg-gray-800 transition-colors duration-200"
              >
                Register
              </Link>
            </>
          )}
        </div>
      </div>
    </nav>
  );
}