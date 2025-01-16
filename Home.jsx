import React , { useContext } from 'react';
import { UserContext } from '../context/user.context';
import User from '../../../backend/models/user.model';

const Home = () => {

    const { user } = useContext(UserContext)

    return (
        <div>{user}</div>
    )
}

export default Home