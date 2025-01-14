import { createBrowserRouter } from "react-router-dom";
import MainPage from "../pages/mainPage/mainPage";
import ErrorPage from "../pages/errorPage/errorPage";
import FontPage from "../pages/fontpage/fontPage";

export const router = createBrowserRouter([
  {
    path: "/",
    element: <MainPage />,
    errorElement: <ErrorPage />,
    children: [
      {
        path: "font",
        element: <FontPage />,
      },
    ],
  },
]);
