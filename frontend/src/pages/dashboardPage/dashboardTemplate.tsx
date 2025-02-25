import { Outlet } from "react-router-dom";
import Header from "../../components/common/Header/Header";
import Footer from "../../components/common/Footer/Footer";

const dashboardTemplate = () => {
  return (
    <div className="min-h-screen flex flex-col">
      <Header />
      <div className="flex-1 bg-red-50 mt-8">
        <Outlet />
      </div>
      <Footer />
    </div>
  );
};

export default dashboardTemplate;
