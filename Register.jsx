import React , { useState } from 'react';
import { Link , useNavigate } from 'react-router-dom';
import axios from '../config/axios';
import Home from './Home';
import Login from './Login';

const Register = () => {

    const [ email , setEmail ] = useState('')
    const [ password , setPassword ] = useState('')
 
    const navigate = useNavigate()


    function submitHandler(e) {
        
         
        axios.post('/users/register', {
            email,
            password
        }).then( (res) => {
            console.log(res.data)
            navigate('/')
        }).catch( (err) => {
            console.log(err.response.data )
        })
    }
    return (
        <div className="min-h-screen bg-gray-900 flex items-center justify-center">
            <div className="bg-gray-800 p-10 rounded-lg shadow-lg max-w-md w-full">  {/* Adjusted width and max-width for rectangular shape */}
                <h2 className="text-xl text-white font-semibold mb-5">Register</h2>  {/* Removed text-center for left alignment */}
                <form 
                    onSubmit={submitHandler} 
                 > 
                    <div className="mb-4">
                        <label htmlFor="email" className="block text-gray-400 mb-2">Email</label>
                        <input 
                               onChange={(e) => setEmail (e.target.value)}
                               type="email" id="email" name="email"
                               className="bg-gray-700 text-white rounded-lg w-full p-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                               placeholder="Enter your email" />
                    </div>
                    <div className="mb-6">
                        <label htmlFor="password" className="block text-gray-400 mb-2">Password</label>
                        <input 
                               onChange={(e) => setPassword(e.target.value)}
                               type="password" id="password" name="password"
                               className="bg-gray-700 text-white rounded-lg w-full p-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
                               placeholder="Enter your password" />
                    </div>
                    <button type="submit" className="w-full bg-blue-500 hover:bg-blue-600 text-white p-3 rounded-lg transition-colors duration-200">
                        Register
                    </button>
                    <div className="mt-4 text-center">
                    <span className="text-gray-400">Already have an account?</span> <Link to="/login" className="text-blue-300 hover:underline">login</Link>
                    </div>

                </form>
            </div>
        </div>
    );
}

export default Register;
