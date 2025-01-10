@Configuration
class OpcuaConfig {
    @Bean
    fun opcuaClient(): OpcUaClient {
        val endpoint = "opc.tcp://localhost:4840"
        return OpcUaClient.create(endpoint)
    }
}