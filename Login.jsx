import React , { useState , useContext } from 'react';
import { Link , useNavigate } from 'react-router-dom';
import axios from '../config/axios';
import { UserContext } from '../context/user.context';


const Login = () => {

    const [ email , setEmail ] = useState('')
    const [ password , setPassword ] = useState('')

    const { setUser } = useContext(UserContext);
    const navigate = useNavigate()


    function submitHandler(e) {

        e.preventDefault()
        axios.post('/users/login', {
            email,
            password
        }).then( (res) => {
            console.log(res.data)

            localStorage.setItem('token' , res.data.token)
            setUser(res.data.user)
            navigate('/')
        }).catch( (err) => {
            console.log(err.response.data )
        })
    }


    return (
        <div className="min-h-screen bg-gray-900 flex items-center justify-center">
            <div className="bg-gray-800 p-10 rounded-lg shadow-lg max-w-md w-full">  {/* Adjusted width and max-width for rectangular shape */}
                <h2 className="text-xl text-white font-semibold mb-5">Login</h2>  {/* Removed text-center for left alignment */}
                <form
                    onSubmit={submitHandler}
                >
                    <div className="mb-4">
                        <label htmlFor="email" className="block text-gray-400 mb-2">Email</label>
                        <input 
                               onChange={(e) => setEmail(e.target.value)}
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
                        Login
                    </button>
                    <div className="mt-4 text-center">
                    <span className="text-gray-400">Don't have an account?</span> <Link to="/register" className="text-blue-300 hover:underline">Create One</Link>

                    </div>

                </form>
            </div>
        </div>
    );
}

export default Login;
