import Home from './pages/Home';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { Toaster } from 'react-hot-toast';
import './App.css';

function App() {
  const queryClient = new QueryClient();
  return (
    <QueryClientProvider client={queryClient}>
      <div className="App">
        <Home></Home>
        <Toaster></Toaster>
      </div>
    </QueryClientProvider>
  );
}

export default App;
