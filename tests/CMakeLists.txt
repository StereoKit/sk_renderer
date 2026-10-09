# Host only: no GPU, no window, so these run anywhere ctest does.

add_executable(skr_tlsf_test
	tlsf_test.c
	../sk_renderer/vk/skr_tlsf.c)
skr_target_warnings(skr_tlsf_test)
add_test(NAME skr_tlsf COMMAND skr_tlsf_test)
