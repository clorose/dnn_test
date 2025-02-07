// path: test/backend/opcua/src/main/kotlin/com/cnc/opcua/OpcuaApplication.kt
package com.cnc.opcua

import org.springframework.boot.autoconfigure.SpringBootApplication
import org.springframework.boot.runApplication

@SpringBootApplication
class OpcuaApplication

fun main(args: Array<String>) {
    runApplication<OpcuaApplication>(*args)
}