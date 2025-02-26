import { useMachineData } from "../../../pages/graphPage/hooks/useMachineData";

const Header = () => {
  const isConnected = useMachineData();
  return (
    <header className="fixed top-0 left-0 w-full bg-green-50 z-50 h-12 flex justify-between items-center px-4">
      <h1 className="text-2xl font-bold">Machine Dashboard</h1>
      <span className={`px-3 py-1 rounded ${isConnected ? "bg-green-500" : "bg-red-500"} text-white`}>
        {isConnected ? "Connected" : "Disconnected"}
      </span>
    </header>
  );
};

export default Header;
