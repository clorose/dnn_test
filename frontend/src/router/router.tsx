import { createBrowserRouter, Navigate } from "react-router-dom";
import ErrorPage from "../pages/errorPage/errorPage";
import GraphPage from "../pages/graphPage/graphPage";
import DashboardPage from "../pages/dashboardPage/dashboardPage";
import DashboardTemplate from "../pages/dashboardPage/dashboardTemplate";

export const router = createBrowserRouter([
  {
    path: "/graph",
    element: <GraphPage />,
    errorElement: <ErrorPage />,
  },
  {
    path: "/",
    element: <DashboardTemplate />,
    errorElement: <ErrorPage />,
    children: [
      {
        index: true,
        element: <Navigate to="/dashboard" replace />,
      },
      {
        path: "dashboard",
        element: <DashboardPage />,
      },
    ],
  },
]);
