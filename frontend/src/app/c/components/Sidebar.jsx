import {
	Sidebar,
	SidebarContent,
	SidebarFooter,
	SidebarGroup,
	SidebarHeader,
 } from "@/components/ui/sidebar"

 export function ChatSideBar() {
	return (
	  <Sidebar>
		 <SidebarHeader />
		 <SidebarContent>
			<SidebarGroup />
			<SidebarGroup />
		 </SidebarContent>
		 <SidebarFooter />
	  </Sidebar>
	)
 }
