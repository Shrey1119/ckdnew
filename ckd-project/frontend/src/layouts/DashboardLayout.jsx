import Sidebar from '../components/Sidebar'
import Navbar from '../components/Navbar'

export default function DashboardLayout({ children, title }) {
  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <div className="flex-1 flex flex-col min-w-0">
        <Navbar title={title} />
        <main className="flex-1 p-6 overflow-auto animate-fade-in">
          {children}
        </main>
      </div>
    </div>
  )
}
