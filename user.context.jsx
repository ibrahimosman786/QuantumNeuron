import React, { createContext, useContext, useState, useEffect } from 'react';

// Create Context Object
export const UserContext = createContext();

// Create a provider for components to consume and subscribe to changes
export const UserProvider = ({ children }) => {
    const [user, setUser] = useState(null);
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        // Simulate fetching user from localStorage or API at startup
        const fetchUser = () => {
            const savedUser = localStorage.getItem('user');
            const token = localStorage.getItem('token');
            if (savedUser && token) {
                setUser(JSON.parse(savedUser));
            }
            setIsLoading(false);
        };

        fetchUser();
    }, []);

    const login = (userData, token) => {
        localStorage.setItem('user', JSON.stringify(userData));
        localStorage.setItem('token', token);
        setUser(userData);
    };

    const logout = () => {
        localStorage.removeItem('user');
        localStorage.removeItem('token');
        setUser(null);
    };

    // Value provided to consuming components
    const value = {
        user,
        isLoading,
        login,
        logout
    };

    return (
        <UserContext.Provider value={value}>
            {!isLoading ? children : <div>Loading...</div>}
        </UserContext.Provider>
    );
};

