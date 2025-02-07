import { createBrowserRouter } from "react-router-dom";
import MainPage from "../pages/mainPage/mainPage";
import ErrorPage from "../pages/errorPage/errorPage";
import FontPage from "../pages/fontpage/fontPage";
import GraphPage from "../pages/graphPage/graphPage";

export const router = createBrowserRouter([
  {
    path: "/",
    element: <MainPage />,
    errorElement: <ErrorPage />,
  },
  {
    path: "/graph",
    element: <GraphPage />,
    errorElement: <ErrorPage />,
  },
  {
    path: "/font",
    element: <FontPage />,
    errorElement: <ErrorPage />,
  }
]);
