// path: backend/predictor/src/main/kotlin/com/cnc/predictor/config/OpcUaClientConfig.kt
package com.cnc.predictor.config

import com.cnc.opcua.client.OpcUaClient
import org.springframework.context.annotation.Bean
import org.springframework.context.annotation.ComponentScan
import org.springframework.context.annotation.Configuration
import org.springframework.context.annotation.Import

@Configuration
@ComponentScan("com.cnc.opcua")
class OpcUaClientConfig {
    
    @Bean
    fun opcUaClient(): OpcUaClient {
        val opcUaClient = OpcUaClient()
        opcUaClient.connect()
        return opcUaClient
    }
}